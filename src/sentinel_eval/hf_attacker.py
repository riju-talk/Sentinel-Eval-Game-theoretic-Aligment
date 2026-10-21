"""Local HuggingFace attacker (base model + optional LoRA adapter from scripts/train_attacker_grpo.py)."""
from __future__ import annotations

from .attacks import ATTACKER_SYSTEM


class HFAttacker:
    def __init__(self, base: str, lora: str | None = None, max_new_tokens: int = 96, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch, self.max_new = torch, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, device_map="cuda:0",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16))
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.model.eval()
        torch.manual_seed(seed)

    def attack(self, canary: str) -> str:
        msgs = [{"role": "system", "content": ATTACKER_SYSTEM}, {"role": "user", "content": f"Token: {canary}"}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new, do_sample=True, temperature=0.8, top_p=0.95)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return text if canary in text else f"{text} {canary}"
