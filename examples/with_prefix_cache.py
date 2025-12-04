# examples/with_prefix_cache.py
import sys
from pathlib import Path

# Add parent directory to path so we can import context_cache
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from context_cache.model_wrapper import HFModelWrapper

def run():
    model = HFModelWrapper("gpt2")

    prefix = "You are a helpful assistant that answers in short, clear sentences.\nUser: "
    question = "Explain what KV caching is in simple terms.\nAssistant:"
    prompt = prefix + question

    # First call: cache miss
    t0 = time.perf_counter()
    out1 = model.generate_with_prefix_cache(prompt, max_new_tokens=64)
    t1 = time.perf_counter()
    print("First call output:\n", out1)
    print(f"\nFirst call latency (cache miss): {(t1 - t0) * 1000:.1f} ms")

    # Second call: cache hit
    t2 = time.perf_counter()
    out2 = model.generate_with_prefix_cache(prompt, max_new_tokens=64)
    t3 = time.perf_counter()
    print("\nSecond call output:\n", out2)
    print(f"\nSecond call latency (cache HIT): {(t3 - t2) * 1000:.1f} ms")

if __name__ == "__main__":
    run()
