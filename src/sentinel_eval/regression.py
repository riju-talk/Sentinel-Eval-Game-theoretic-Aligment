"""Turn diagnosed failures into permanent regression cases and replay them against a new agent version in CI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .agent import ReActAgent
from .ehr import EHR, build_ehr
from .replay import CausalFailureGraph
from .trace import Trace

DEFAULT_PATH = Path("regressions/cases.jsonl")


def export_case(trace: Trace, graph: CausalFailureGraph | None, ehr_seed: int, n_patients: int,
                anomaly_rate: float = 0.0, mode: str = "other", path: Path = DEFAULT_PATH) -> dict:
    case = {"task_id": trace.task_id, "ehr_seed": ehr_seed, "n_patients": n_patients, "anomaly_rate": anomaly_rate,
            "mode": mode, "root_step": graph.root_step if graph else None, "gold": trace.gold,
            "bad_answer": trace.final_answer}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(case) + "\n")
    return case


def load_cases(path: Path = DEFAULT_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def run_regressions(make_agent: Callable[[EHR], ReActAgent], path: Path = DEFAULT_PATH) -> list[dict]:
    """Re-run every recorded failure; a case 'regresses' if the agent still (or again) gets it wrong."""
    results = []
    for c in load_cases(path):
        ehr = build_ehr(c["n_patients"], c["ehr_seed"], c["anomaly_rate"])
        task = next(t for t in ehr.tasks if t.id == c["task_id"])
        trace = make_agent(ehr).run(task)
        results.append({**c, "passed": bool(trace.success), "answer": trace.final_answer})
    return results
