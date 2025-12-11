# benchmarks/benchmark_prefix_reuse.py

import time
from statistics import mean
from context_cache.model_wrapper import HFModelWrapper


def benchmark_no_cache(
    model: HFModelWrapper,
    prefix: str,
    deltas: list[str],
    num_queries: int,
    max_new_tokens: int,
) -> list[float]:
    """
    Run num_queries full prompts (prefix + delta) with generate_no_cache
    and return a list of latencies in milliseconds.
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

        lat_ms = (t1 - t0) * 1000.0
        latencies_ms.append(lat_ms)

    return latencies_ms


def benchmark_with_prefix_cache(
    model: HFModelWrapper,
    prefix: str,
    deltas: list[str],
    num_queries: int,
    max_new_tokens: int,
) -> list[float]:
    """
    Run num_queries queries with shared prefix using generate_with_reuse.
    The prefix KVs are cached once and reused across queries.
    Returns a list of latencies in milliseconds.
    """
    latencies_ms: list[float] = []

    # Pre-warm: compute and cache KVs for the prefix (not timed)
    model.get_past_for_prefix(prefix)

    for i in range(num_queries):
        delta = deltas[i % len(deltas)]

        t0 = time.perf_counter()
        _ = model.generate_with_reuse(prefix, delta, max_new_tokens=max_new_tokens)
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000.0
        latencies_ms.append(lat_ms)

    return latencies_ms


def summarize(latencies_ms: list[float]) -> tuple[float, float, float]:
    """
    Return (avg, min, max) in milliseconds.
    """
    if not latencies_ms:
        return 0.0, 0.0, 0.0
    return (
        mean(latencies_ms),
        min(latencies_ms),
        max(latencies_ms),
    )


def run_benchmark(
    model_id: str = "gpt2",
    num_queries: int = 10,
    max_new_tokens: int = 32,
):
    print(f"Loading model/tokenizer: {model_id} ...")
    model = HFModelWrapper(model_id=model_id)

    # Shared prefix and a small set of different questions (deltas)
    base_prefix = (
    "You are a helpful assistant that answers in short, clear sentences. "
    "We are discussing KV caching in large language models. "
    "You will see several questions about caching, transformers, and performance. "
)

    # Repeat to create a long prefix (e.g., ~1000+ tokens depending on model tokenization)
    PREFIX_REPEATS = 20  # try 20, 40, etc
    prefix = (base_prefix + "\n") * PREFIX_REPEATS

    deltas = [
        "User: Explain what KV caching is in simple terms.\nAssistant:",
        "User: Why is KV caching useful for large language models?\nAssistant:",
        "User: How does KV caching speed up repeated queries?\nAssistant:",
        "User: When would KV caching NOT help much?\nAssistant:",
    ]

    print(f"Running benchmark with model={model_id}, num_queries={num_queries}, "
          f"max_new_tokens={max_new_tokens}\n")

    # --- No-cache baseline ---
    no_cache_lat = benchmark_no_cache(
        model=model,
        prefix=prefix,
        deltas=deltas,
        num_queries=num_queries,
        max_new_tokens=max_new_tokens,
    )
    no_cache_avg, no_cache_min, no_cache_max = summarize(no_cache_lat)

    # --- With prefix-cache ---
    cache_lat = benchmark_with_prefix_cache(
        model=model,
        prefix=prefix,
        deltas=deltas,
        num_queries=num_queries,
        max_new_tokens=max_new_tokens,
    )
    cache_avg, cache_min, cache_max = summarize(cache_lat)

    # --- Optional: compute speedup ---
    speedup = (no_cache_avg / cache_avg) if cache_avg > 0 else 0.0

    # --- Print detailed results (per-mode) ---
    print("No-cache latencies (ms):")
    print(", ".join(f"{x:.1f}" for x in no_cache_lat))
    print("\nWith-prefix-cache latencies (ms):")
    print(", ".join(f"{x:.1f}" for x in cache_lat))

    # --- Print README-ready markdown table ---
    print("\n\nMarkdown summary table (paste into README):\n")
    print("| Mode               | N   | Avg latency (ms) | Min (ms) | Max (ms) |")
    print("|--------------------|-----|------------------|----------|----------|")
    print(
        f"| No cache           | {num_queries:<3} | {no_cache_avg:>16.1f} | "
        f"{no_cache_min:>8.1f} | {no_cache_max:>8.1f} |"
    )
    print(
        f"| With prefix cache  | {num_queries:<3} | {cache_avg:>16.1f} | "
        f"{cache_min:>8.1f} | {cache_max:>8.1f} |"
    )
    print(f"\nEstimated average speedup: **{speedup:.2f}×**")


if __name__ == "__main__":
    run_benchmark()
