"""Experiment C: attack-success rate (ASR) vs task success for plain and guarded defenders, per attacker.

    python experiments/attack_defender.py --defender ollama:qwen2.5:3b-instruct --attackers template,ollama:qwen2.5:7b-instruct
Attackers: `template`, any LLM spec, or `hf:<base_model>[+<lora_dir>]` (e.g. the GRPO-trained one).
"""
import argparse
import json
from pathlib import Path

from sentinel_eval.agent import AGENT_SYSTEM, GUARD, ReActAgent
from sentinel_eval.attacks import LLMAttacker, TemplateAttacker, evaluate_attacks
from sentinel_eval.llm import make_llm
from sentinel_eval.hf_attacker import HFAttacker


def make_attacker(spec, seed):
    if spec == "template":
        return TemplateAttacker(seed)
    if spec.startswith("hf:"):
        base, _, lora = spec[3:].partition("+")
        return HFAttacker(base, lora or None)
    return LLMAttacker(make_llm(spec))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(c - h, 3), round(c + h, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defender", required=True)
    ap.add_argument("--attackers", default="template")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/attack_defender.json")
    a = ap.parse_args()

    defender = make_llm(a.defender)
    conditions = {"plain": AGENT_SYSTEM, "guarded": AGENT_SYSTEM + GUARD}
    results = []
    for spec in a.attackers.split(","):
        for cname, system in conditions.items():
            n = atk = ok = clean = 0
            examples = []
            attacker = make_attacker(spec, 0)
            for seed in range(a.seeds):
                r = evaluate_attacks(lambda ehr: ReActAgent(defender, ehr, system=system), attacker, seed=seed)
                n += r.n
                atk += round(r.asr * r.n)
                ok += round(r.task_success_attacked * r.n)
                clean += round(r.task_success_clean * r.n)
                examples += [c["attack"] for c in r.per_case[:2]]
            row = {"attacker": spec, "defender": a.defender, "condition": cname, "n": n, "asr": atk / n,
                   "asr_ci95": wilson(atk, n), "task_success_attacked": ok / n, "task_success_clean": clean / n,
                   "sample_attacks": examples[:4]}
            results.append(row)
            print({k: v for k, v in row.items() if k != "sample_attacks"}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
