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

    # Warm-up
    print("Warming up ...")
    _=model.generate_no_cache(prompt, max_new_tokens=64)

    t0 = time.perf_counter()
    out = model.generate_no_cache(prompt, max_new_tokens=64)
    t1 = time.perf_counter()

    print("Output:\n", out)
    print(f"\nLatency (no prefix reuse): {(t1 - t0) * 1000:.1f} ms")

if __name__ == "__main__":
    run()