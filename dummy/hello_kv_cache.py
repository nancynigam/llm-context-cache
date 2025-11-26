import time
import torch
# Importing 2 classes from Hugging Face's transformers library:
# AutoModelForCausalLM: for loading pre-trained causal language models (e.g., GPT-2);Causal LM: models that predict the next token based on the tokens to the left.
# AutoTokenizer: for tokenizing text into input IDs and managing tokenization operations
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "gpt2"  # small and fast; switch later to a larger model
PREFIX = "Explain tranformers in simple terms."
DELTA = "Include a short example."

def t():
    # returns a high-resolution timestamp (in seconds) ideal for benchmarking code execution time.
    return time.perf_counter()

def secs(x):
    return f"{x*1000:.1f} ms"

def load_engine(model_id=MODEL_ID):
    
    # load the tokenizer for the model; loads the tokenizer configuration and vocab files for the model
    tok = AutoTokenizer.from_pretrained(model_id)
    # GPT-2 has no pad token; 
    # To still process multiple sentences of different lengths together (batching), we need padding since it's a rectangular matrix.
    # To handle this, whenever padding is needed, the tokenizer inserts <eos> instead.
    tok.pad_token = tok.eos_token

    # For causal LMs like GPT-2, the model predicts the next token based only on tokens to the left.
    # That’s why we pad on the left — so that the real text stays aligned to the right, and the model focuses on recent tokens.
    tok.padding_side = "left"

    # Use cpu for now, if you've GPU: device_map="auto", torch_dtype=torch.float16 for faster inference.
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    model.eval()
    return tok, model, device

def encode(tok, device, text):
    enc = tok(text, return_tensors="pt").to(device)
    return {k: v.to(device) for k, v in enc.items()}

# Baseline testing (no reuse)
# Feed the full prompt and measure how long the prefill takes for generating the first token.
@torch.inference_mode()
def baseline_generate(tok, model, device, prompt, max_new_tokens=32):
    inputs = encode(tok, device, prompt)
    t0 = t()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
    do_sample=False,
    use_cache=True)
    dt = t() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    return text, dt

@torch.inference_mode()
def prefill_prefix(tok, model, device, prefix_text):
    enc = encode(tok, device, prefix_text)
    # Forward pass to get PKV (Past Key-Value) for the prefix
    out = model(
        input_ids=enc['input_ids'],
        attention_mask=enc["attention_mask"],
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False
    )
    past_key_values = out.past_key_values
    # We also keep the attention mask length so we can append delta later
    prefix_len = enc["input_ids"].shape[-1]
    return past_key_values, prefix_len


@torch.inference_mode()
def generate_with_pkv(tok, model, device, past_key_values, prefix_len, delta_text, max_new_tokens=32):
    # Encode only the delta
    delta = encode(tok, device, delta_text)

    t0 = t()
    delta_mask = delta.get("attention_mask")
    cache_position = torch.arange(
        prefix_len,
        prefix_len + delta["input_ids"].shape[-1],
        device=device,
        dtype=torch.long,
    )

    # NOTE: we re-use PKV and only feed delta tokens as current inputs.
    out = model.generate(
        input_ids=delta["input_ids"],
        attention_mask=delta_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    dt = t() - t0

    text = tok.decode(out[0], skip_special_tokens=True)
    return text, dt


def main():
    tok, model, device = load_engine()

    full_prompt = PREFIX + DELTA

    print("=== Baseline (no reuse) ===")
    text_base, dt_base = baseline_generate(tok, model, device, full_prompt, max_new_tokens=32)
    print("Baseline time:", secs(dt_base))

    print("\n=== KV Cache path ===")
    # 1) prefill on prefix to capture PKV
    t0 = t()
    pkv, prefix_len = prefill_prefix(tok, model, device, PREFIX)
    dt_prefill = t() - t0
    print("Prefill (prefix only):", secs(dt_prefill))

    # 2) generate with cached PKV and only the delta
    text_kv, dt_kv = generate_with_pkv(tok, model, device, pkv, prefix_len, DELTA, max_new_tokens=32)
    print("Decode w/ PKV (delta only):", secs(dt_kv))

    # Show a quick comparison
    print("\n=== Comparison ===")
    print(f"Baseline end-to-end (prefix+delta in one go): {secs(dt_base)}")
    print(f"KV path: prefill(prefix) {secs(dt_prefill)} + decode(delta) {secs(dt_kv)}")
    print("Tip: On repeated requests with the SAME prefix, you can SKIP prefill entirely and reuse PKV directly.")


if __name__ == "__main__":
    main()