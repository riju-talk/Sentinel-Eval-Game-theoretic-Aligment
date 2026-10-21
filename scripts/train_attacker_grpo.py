"""Single-turn attacker training with GRPO (TRL) + QLoRA.  NOT run in CI; needs a GPU and a defender endpoint.

The attacker writes an injection that is appended to every clinical note; the reward is whether the *defender*
(any OpenAI-compatible endpoint, e.g. `ollama serve`) outputs the attacker's canary instead of the true answer.

    python scripts/train_attacker_grpo.py --defender ollama:qwen2.5:7b-instruct --model Qwen/Qwen2.5-1.5B-Instruct

Sized for an 8 GB GPU (4-bit base model, LoRA r=16, 4 generations per prompt).
"""
import argparse

from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from sentinel_eval.agent import ReActAgent
from sentinel_eval.attacks import ATTACKER_SYSTEM, attacker_reward
from sentinel_eval.ehr import build_ehr, poison_notes
from sentinel_eval.llm import make_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--defender", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="runs/attacker_grpo")
    a = ap.parse_args()

    defender = make_llm(a.defender)
    prompts = [{"prompt": [{"role": "system", "content": ATTACKER_SYSTEM},
                           {"role": "user", "content": f"Token: CANARY-{i:04d}"}], "canary": f"CANARY-{i:04d}"}
               for i in range(512)]

    def reward(prompts, completions, canary, **_):
        rewards = []
        for comp, c in zip(completions, canary):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            ehr = build_ehr(6, seed=hash(c) % 1000)
            poison_notes(ehr, text)
            task = next(t for t in ehr.tasks if t.kind == "note")
            tr = ReActAgent(defender, ehr).run(task, injection_markers=(c,))
            rewards.append(attacker_reward(text, c, bool(tr.meta["attack_success"])))
        return rewards

    cfg = GRPOConfig(output_dir=a.out, per_device_train_batch_size=4, num_generations=4, max_completion_length=96,
                     learning_rate=5e-6, max_steps=a.steps, logging_steps=5, bf16=True,
                     gradient_accumulation_steps=2, report_to="none",
                     model_init_kwargs={"quantization_config": BitsAndBytesConfig(
                         load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")})
    GRPOTrainer(model=a.model, reward_funcs=reward, args=cfg, train_dataset=Dataset.from_list(prompts),
                peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")).train()


if __name__ == "__main__":
    main()
