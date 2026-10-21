"""Who&When (ICML 2025) loader and evaluator: agent-level and step-level failure-attribution accuracy."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import pandas as pd

from .attribution import Attribution
from .trace import Trace

DEFAULT_DIR = Path("data/raw/who_and_when")
SUBSETS = {"hand": "Hand-Crafted", "algo": "Algorithm-Generated"}
URL = "https://huggingface.co/datasets/Kevin355/Who_and_When/resolve/main/{}.parquet"


def download(dest: Path = DEFAULT_DIR) -> None:
    import httpx
    dest.mkdir(parents=True, exist_ok=True)
    for name in SUBSETS.values():
        p = dest / f"{name}.parquet"
        if not p.exists():
            p.write_bytes(httpx.get(URL.format(name), follow_redirects=True, timeout=120).content)


def _agent_of(h: dict) -> str:
    # Hand-Crafted logs carry the agent in `role`; algorithm-generated logs in `name`.
    return str(h.get("name") or h.get("role") or "unknown")


def load(subset: str = "hand", data_dir: Path = DEFAULT_DIR) -> list[Trace]:
    p = Path(data_dir) / f"{SUBSETS[subset]}.parquet"
    if not p.exists():
        download(Path(data_dir))
    traces = []
    for i, r in pd.read_parquet(p).iterrows():
        t = Trace(task_id=str(r.get("question_ID", i)), task=str(r["question"]), gold=str(r.get("groundtruth", r.get("ground_truth"))),
                  success=False,
                  meta={"mistake_agent": str(r["mistake_agent"]), "mistake_step": int(r["mistake_step"]),
                        "mistake_reason": str(r["mistake_reason"]), "subset": subset})
        for j, h in enumerate(r["history"]):
            kind = "input" if h.get("role") == "human" else "message"
            t.add(kind, str(h.get("content", "")), agent=_agent_of(h))
        traces.append(t)
    return traces


Attributor = Callable[[Trace], Attribution]


def evaluate(traces: list[Trace], attributor: Attributor) -> dict:
    n = a_ok = s_ok = 0
    per_task = []
    for t in traces:
        pred = attributor(t)
        a = pred.agent is not None and pred.agent.strip().lower() == t.meta["mistake_agent"].strip().lower()
        s = pred.step == t.meta["mistake_step"]
        n, a_ok, s_ok = n + 1, a_ok + a, s_ok + s
        per_task.append({"task_id": t.task_id, "agent_ok": a, "step_ok": s, "pred": pred.__dict__ | {"mode": str(pred.mode)}})
    return {"n": n, "agent_acc": a_ok / max(n, 1), "step_acc": s_ok / max(n, 1), "per_task": per_task}


def ensemble(runs: list[dict], traces: list[Trace]) -> dict:
    """Majority vote over per-judge result files (each from `evaluate`, same trace order). Ties go to the first judge."""
    from collections import Counter

    def vote(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        top = Counter(vals).most_common()
        return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else vals[0]

    n = a_ok = s_ok = 0
    for i, t in enumerate(traces[: len(runs[0]["per_task"])]):
        preds = [r["per_task"][i]["pred"] for r in runs]
        agent, step = vote([p["agent"] for p in preds]), vote([p["step"] for p in preds])
        n += 1
        a_ok += agent is not None and agent.lower() == t.meta["mistake_agent"].lower()
        s_ok += step == t.meta["mistake_step"]
    return {"n": n, "agent_acc": a_ok / n, "step_acc": s_ok / n, "judges": len(runs)}


def bootstrap_ci(flags: list[bool], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(flags)
    ms = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters)]


def random_attributor(seed: int = 0) -> Attributor:
    """Uniform random agent (from agents in the log) and step (from non-human steps): the floor to beat."""
    rng = random.Random(seed)

    def f(t: Trace) -> Attribution:
        idxs = [s.idx for s in t.steps if s.kind != "input"]
        step = rng.choice(idxs)
        return Attribution(t.steps[step].agent, step, method="random")
    return f


def expected_random(traces: list[Trace]) -> dict:
    """Closed-form expectation of the random baseline (no sampling noise)."""
    a = s = 0.0
    for t in traces:
        idxs = [x.idx for x in t.steps if x.kind != "input"]
        a += sum(t.steps[i].agent.lower() == t.meta["mistake_agent"].lower() for i in idxs) / len(idxs)
        s += (t.meta["mistake_step"] in idxs) / len(idxs)
    return {"n": len(traces), "agent_acc": a / len(traces), "step_acc": s / len(traces)}
