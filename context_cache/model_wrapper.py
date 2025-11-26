from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HFModelWrapper:
    def __init__(self, model_id: str = "gpt2", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()

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
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)