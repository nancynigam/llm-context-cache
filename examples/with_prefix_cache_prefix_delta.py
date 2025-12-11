# examples/with_prefix_cache_prefix_delta.py

import time
import sys
from pathlib import Path

# Add parent directory to path so we can import context_cache
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_cache.model_wrapper import HFModelWrapper


def run():
    model = HFModelWrapper("gpt2")

    prefix = (
        "You are a helpful assistant that answers in short, clear sentences.\n"
        "User: I will ask you about KV caching.\n"
        "Assistant: Sure, I can help.\n"
    )

    q1 = "User: Explain what KV caching is in simple terms.\nAssistant:"
    q2 = "User: Why is KV caching useful for large language models?\nAssistant:"

    # First question: prefix MISS, compute and cache
    t0 = time.perf_counter()
    out1 = model.generate_with_reuse(prefix, q1, max_new_tokens=50)
    t1 = time.perf_counter()
    print("Q1 answer:\n", out1)
    print(f"\nQ1 latency (prefix MISS): {(t1 - t0) * 1000:.1f} ms")

    # Second question: prefix HIT, skip re-reading prefix
    t2 = time.perf_counter()
    out2 = model.generate_with_reuse(prefix, q2, max_new_tokens=50)
    t3 = time.perf_counter()
    print("\nQ2 answer:\n", out2)
    print(f"\nQ2 latency (prefix HIT): {(t3 - t2) * 1000:.1f} ms")

if __name__ == "__main__":
    run()
