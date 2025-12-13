from typing import Any, List, Optional, Tuple
from typing import Hashable
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import copy
from context_cache.cache import SimpleKVCache
from enum import Enum

class CacheMode(str, Enum):
    NONE = "none"     # recompute prefix everytime, no caching
    EXACT = "exact"   # exact-prefix KV cache
    TRIE = "trie"     # new longest-prefix trie cache

class HFModelWrapper:
    def __init__(self, model_id: str = "gpt2", device: str = None,
                 cache_mode: CacheMode = CacheMode.EXACT):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
       
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        # feature flag
        self.cache_mode = cache_mode

        # exact-prefix cache (what you already had)
        self.exact_cache = SimpleKVCache(max_size=128)

        # trie cache placeholder; we’ll implement / import later
        self.trie_cache = None  # will define below


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

    # Exact match prefix
    def get_past_for_prefix_exact(self, prefix: str):
        """Return (past_key_values, input_ids) from cache or compute & store."""
        prefix_inputs = self.tokenize(prefix)
        # output of tokenize is inputs = {
        # 'input_ids': tensor([[2014, 389, 257, 12947, 3303, ..., 22557, 25]]),  # Shape: [1, 25]
        # 'attention_mask': tensor([[1, 1, 1, 1, 1, ..., 1, 1]])  # Shape: [1, 25]}

        # Attention masks allow us to send a batch into the transformer even when the examples in the batch have varying lengths.
        # attention mask is a binary tensor that tells the model which tokens to pay attention to and which to ignore, particularly 
        # in batches of sequences with padding tokens. A value of 1 indicates a token that the model should attend to, while a 0 indicates 
        # a token that should be ignored, such as a padding token added to make all sequences the same length
        prefix_ids = prefix_inputs["input_ids"]
        key = HFModelWrapper.tensor_key(prefix_ids)

        cached = self.exact_cache.get(key)
        if cached is not None:
            print("[EXACT CACHE] HIT for prefix")
            return copy.deepcopy(cached), prefix_ids

        print("[EXACT CACHE] MISS for prefix")
        with torch.no_grad():
            outputs = self.model(
                **prefix_inputs, # tracks input_ids & attention_mask
                use_cache=True,
            )
        past = outputs.past_key_values # Extracts the past_key_values (KV cache) from the model output
        self.exact_cache.put(key, copy.deepcopy(past))
        return past, prefix_ids
    
    def _get_past_for_prefix(self, prefix: str):
        """
        Internal helper that routes based on cache_mode.
        Returns (past+kv, prefix_input_ids).
        """

        if self.cache_mode == CacheMode.NONE:
            inputs = self.tokenize(prefix)
            with torch.no_grad():
                outputs = self.mode(**inputs, use_cache=True)
            return outputs.past_key_values, inputs["input_ids"]
        
        elif self.cache_mode == CacheMode.EXACT:
            return self.get_past_for_prefix_exact(prefix)

        elif self.cache_mode == CacheMode.TRIE:
            # placeholder for when you add trie logic
            # for now, just fall back to exact so things still work
            print("[TRIE] mode not implemented yet, using exact as fallback")
            return self.get_past_for_prefix_exact(prefix)
        
        else:
            raise ValueError(f"Unknown cache_mode: {self.cache_mode}")


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

    def generate_with_reuse(
        self,
        prefix: str,
        delta: str,
        max_new_tokens: int = 50
    ) -> str:
        """
        1. Get / compute KV cache for prefix.
        2. Extend the context by running delta on top of prefix KVs.
        3. Decode additional tokens from there.
        """

        # 1. Get KVs for prefix
        past_prefix, prefix_ids = self._get_past_for_prefix(prefix)

        # 2. Run delta on top of prefix KVs
        delta_inputs = self.tokenize(delta)
        delta_ids = delta_inputs["input_ids"]

        with torch.no_grad():
            delta_outputs = self.model(
                input_ids = delta_ids,
                past_key_values = past_prefix,
                use_cache = True,
            )

            past_after_delta = delta_outputs.past_key_values
            full_ids = torch.cat([prefix_ids, delta_ids], dim=-1)

        # 3. Decode additional tokens starting from last token of (prefix + delta)
        last_token = full_ids[:, -1:]
        generated = last_token
        past_kv = past_after_delta

        with torch.no_grad():
            for _ in range(max_new_tokens):
                step_outputs = self.model(
                    input_ids = generated[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                )
                logits = step_outputs.logits[:, -1, :]
                past_kv = step_outputs.past_key_values

                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

        # Reconstruct full sequence: prefix + delta + generated (skip duplicate last_token)
        full_seq = torch.cat([full_ids, generated[:, 1:]], dim=-1)
        return self.tokenizer.decode(full_seq[0], skip_special_tokens=True)

        
