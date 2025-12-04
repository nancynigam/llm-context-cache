from typing import Any, List, Optional, Tuple
from typing import Hashable
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from context_cache.cache import SimpleKVCache

class HFModelWrapper:
    def __init__(self, model_id: str = "gpt2", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.prefix_cache = SimpleKVCache(max_size=128)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)


    def tokenize(self, text:str):
        return self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)
    
    # Baseline - no prefix reuse generation path
    def generate_no_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> str:
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True, # still uses internal cache *within* this call
                # use_cache=True only reuses KV within one generation call
                # A real KV cache reuses KV across multiple generation calls (across requests, across users, across sessions).
            )
        # Converts token IDs back to text, output_ids[0]: First (and only) sequence in the batch
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True) # Removes special tokens like <pad>, <eos>
    
    # The conversion to a tuple makes the tensor hashable because tuples in Python are immutable, 
    # whereas lists and PyTorch tensors are generally mutable. 
    # Python's core requirement for an object to be hashable is that it must be immutable.
    @staticmethod
    def tensor_key(input_ids : torch.Tensor) -> Hashable:
        # input_ids: shape [1, seq_len]
        # This helps the cache map token sequneces to their precomputed KV states for reuse
        # PyTorch tensors are not hashable & KV cache needs hashable keys to look up cached KV states.
        return tuple(input_ids[0].tolist()) 

    def get_past_for_prompt(self, prompt: str):
        """Return (past_key_values, input_ids) from cache or compute & store."""
        inputs = self.tokenize(prompt)
        # output of tokenize is inputs = {
        # 'input_ids': tensor([[2014, 389, 257, 12947, 3303, ..., 22557, 25]]),  # Shape: [1, 25]
        # 'attention_mask': tensor([[1, 1, 1, 1, 1, ..., 1, 1]])  # Shape: [1, 25]}

        # Attention masks allow us to send a batch into the transformer even when the examples in the batch have varying lengths.
        # attention mask is a binary tensor that tells the model which tokens to pay attention to and which to ignore, particularly 
        # in batches of sequences with padding tokens. A value of 1 indicates a token that the model should attend to, while a 0 indicates 
        # a token that should be ignored, such as a padding token added to make all sequences the same length
        input_ids = inputs["input_ids"]
        key = HFModelWrapper.tensor_key(input_ids)

        cached = self.prefix_cache.get(key)
        if cached is not None:
            print("[CACHE] HIT")
            return cached, input_ids

        print("[CACHE] MISS")
        with torch.no_grad():
            outputs = self.model(
                **inputs, # tracks input_ids & attention_mask
                use_cache=True,
            )
        past = outputs.past_key_values # Extracts the past_key_values (KV cache) from the model output
        self.prefix_cache.put(key, past)
        return past, input_ids
    
    def generate_with_prefix_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> str:
        """
        MVP: If we've seen this exact prompt before, reuse its past_key_values.
        If not, compute once, cache, then generate.
        """
        past, input_ids = self.get_past_for_prompt(prompt)

        # We want to start generating from the end of the prompt
        # Using the last token as the "current" input to pass along with past
        last_token = input_ids[:, -1:]

        generated = last_token
        past_kv = past

        # Context manager ensures gradients are disabled for forward pass to save speed & memory
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=generated[:, -1:],  # last token
                    past_key_values=past_kv,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_kv = outputs.past_key_values

                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

        # Reconstruct full sequence: prompt_ids + newly generated (excluding duplicated last_token)
        full_ids = torch.cat([input_ids, generated[:, 1:]], dim=-1)
        return self.tokenizer.decode(full_ids[0], skip_special_tokens=True)