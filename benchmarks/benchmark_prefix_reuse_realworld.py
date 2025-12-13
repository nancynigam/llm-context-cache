# benchmarks/benchmark_prefix_reuse.py

import time
from statistics import mean
from context_cache.model_wrapper import HFModelWrapper, CacheMode


def benchmark_no_cache(
    model: HFModelWrapper,
    prefix: str,
    deltas: list[str],
    num_queries: int,
    max_new_tokens: int,
) -> list[float]:
    """
    Baseline: run num_queries full prompts (prefix + delta)
    with generate_no_cache and return per-query latencies in ms.
    """
    latencies_ms: list[float] = []

    # Optional warmup (not timed)
    warm_full = prefix + deltas[0]
    _ = model.generate_no_cache(warm_full, max_new_tokens=max_new_tokens)

    for i in range(num_queries):
        delta = deltas[i % len(deltas)]
        full_prompt = prefix + delta

        t0 = time.perf_counter()
        _ = model.generate_no_cache(full_prompt, max_new_tokens=max_new_tokens)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    return latencies_ms


def benchmark_with_reuse_mode(
    model: HFModelWrapper,
    prefix: str,
    deltas: list[str],
    num_queries: int,
    max_new_tokens: int,
) -> list[float]:
    """
    Generic benchmark for any cache_mode that uses generate_with_reuse.
    This will respect model.cache_mode (NONE / EXACT / TRIE).

    We do a single warmup call (not timed) to allow the cache to populate,
    then measure num_queries calls.
    """
    latencies_ms: list[float] = []

    # Warmup: populate cache once (or just warm the model path)
    _ = model.generate_with_reuse(prefix, deltas[0], max_new_tokens=max_new_tokens)

    for i in range(num_queries):
        delta = deltas[i % len(deltas)]

        t0 = time.perf_counter()
        _ = model.generate_with_reuse(prefix, delta, max_new_tokens=max_new_tokens)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    return latencies_ms


def summarize(latencies_ms: list[float]) -> tuple[float, float, float]:
    """
    Return (avg, min, max) in milliseconds.
    """
    if not latencies_ms:
        return 0.0, 0.0, 0.0
    return mean(latencies_ms), min(latencies_ms), max(latencies_ms)


def run_benchmark(
    model_id: str = "gpt2",
    num_queries: int = 10,
    max_new_tokens: int = 32,
):
    print(f"Loading model/tokenizer: {model_id} ...")

    # Shared prefix and a long-ish context (realistic chat history)
    base_prefix = (
        "You are a helpful assistant that answers in short, clear sentences. "
        "We are discussing KV caching in large language models. "
        "You will see several questions about caching, transformers, and performance. "
    )

    PREFIX_REPEATS = 20  # tweak this to control prefix length
    prefix = (base_prefix + "\n") * PREFIX_REPEATS

    deltas = [
        "User: Explain what KV caching is in simple terms.\nAssistant:",
        "User: Why is KV caching useful for large language models?\nAssistant:",
        "User: How does KV caching speed up repeated queries?\nAssistant:",
        "User: When would KV caching NOT help much?\nAssistant:",
    ]

    print(
        f"Running benchmark with model={model_id}, num_queries={num_queries}, "
        f"max_new_tokens={max_new_tokens}\n"
    )

    # --- Compute prefix token length once for logging/README context ---
    tmp_model_for_tokens = HFModelWrapper(model_id=model_id, cache_mode=CacheMode.NONE)
    prefix_ids = tmp_model_for_tokens.tokenize(prefix)["input_ids"]
    prefix_len_tokens = prefix_ids.shape[1]
    print(f"Shared prefix length: {prefix_len_tokens} tokens\n")

    # --- 1. No-cache baseline (generate_no_cache) ---
    model_no_cache = HFModelWrapper(model_id=model_id, cache_mode=CacheMode.NONE)
    print("[1/3] Benchmarking: NO CACHE (generate_no_cache)...")
    no_cache_lat = benchmark_no_cache(
        model=model_no_cache,
        prefix=prefix,
        deltas=deltas,
        num_queries=num_queries,
        max_new_tokens=max_new_tokens,
    )
    no_cache_avg, no_cache_min, no_cache_max = summarize(no_cache_lat)

    # --- 2. Exact prefix cache mode ---
    model_exact = HFModelWrapper(model_id=model_id, cache_mode=CacheMode.EXACT)
    print("[2/3] Benchmarking: EXACT PREFIX CACHE (CacheMode.EXACT)...")
    exact_lat = benchmark_with_reuse_mode(
        model=model_exact,
        prefix=prefix,
        deltas=deltas,
        num_queries=num_queries,
        max_new_tokens=max_new_tokens,
    )
    exact_avg, exact_min, exact_max = summarize(exact_lat)

    # --- 3. Trie prefix cache mode ---
    model_trie = HFModelWrapper(model_id=model_id, cache_mode=CacheMode.TRIE)
    print("[3/3] Benchmarking: TRIE PREFIX CACHE (CacheMode.TRIE)...")
    trie_lat = benchmark_with_reuse_mode(
        model=model_trie,
        prefix=prefix,
        deltas=deltas,
        num_queries=num_queries,
        max_new_tokens=max_new_tokens,
    )
    trie_avg, trie_min, trie_max = summarize(trie_lat)

    # --- Speedups (vs no-cache) ---
    speedup_exact = (no_cache_avg / exact_avg) if exact_avg > 0 else 0.0
    speedup_trie = (no_cache_avg / trie_avg) if trie_avg > 0 else 0.0

    # --- Print detailed results (per-mode) ---
    print("\nNo-cache latencies (ms):")
    print(", ".join(f"{x:.1f}" for x in no_cache_lat))

    print("\nExact prefix cache latencies (ms):")
    print(", ".join(f"{x:.1f}" for x in exact_lat))

    print("\nTrie prefix cache latencies (ms):")
    print(", ".join(f"{x:.1f}" for x in trie_lat))

    # --- Print README-ready markdown table ---
    print("\n\nMarkdown summary table (paste into README):\n")
    print("| Mode               | N   | Avg latency (ms) | Min (ms) | Max (ms) | Speedup vs No-cache |")
    print("|--------------------|-----|------------------|----------|----------|----------------------|")
    print(
        f"| No cache           | {num_queries:<3} | {no_cache_avg:>16.1f} | "
        f"{no_cache_min:>8.1f} | {no_cache_max:>8.1f} | {'1.00×':>20} |"
    )
    print(
        f"| Exact prefix cache | {num_queries:<3} | {exact_avg:>16.1f} | "
        f"{exact_min:>8.1f} | {exact_max:>8.1f} | {speedup_exact:>6.2f}×{'':>11} |"
    )
    print(
        f"| Trie prefix cache  | {num_queries:<3} | {trie_avg:>16.1f} | "
        f"{trie_min:>8.1f} | {trie_max:>8.1f} | {speedup_trie:>6.2f}×{'':>11} |"
    )

    print(
        f"\nShared prefix length (for this run): **{prefix_len_tokens} tokens** "
        f"(PREFIX_REPEATS={PREFIX_REPEATS})"
    )


if __name__ == "__main__":
    run_benchmark()
