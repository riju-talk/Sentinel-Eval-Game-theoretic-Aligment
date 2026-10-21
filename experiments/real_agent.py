"""Experiment B: a real open-weight LLM agent on the EHR testbed. Failures are diagnosed by (1) counterfactual replay
(intervention) and (2) an LLM jury (attribution), and the two are compared.

    python experiments/real_agent.py --agent ollama:qwen2.5:7b-instruct --judges ollama:llama3.1:8b --anomaly-rate 0.3
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from sentinel_eval.agent import ReActAgent
from sentinel_eval.attribution import all_at_once
from sentinel_eval.ehr import build_ehr
from sentinel_eval.judge import Jury
from sentinel_eval.llm import make_llm
from sentinel_eval.regression import export_case
from sentinel_eval.replay import find_root_cause
from sentinel_eval.trace import save_traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--judges", required=True, help="comma-separated; attribution jury")
    ap.add_argument("--patients", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anomaly-rate", type=float, default=0.0)
    ap.add_argument("--out", default="runs/real_agent")
    a = ap.parse_args()

    out = Path(a.out)
    ehr = build_ehr(a.patients, a.seed, a.anomaly_rate)
    agent = ReActAgent(make_llm(a.agent), ehr)
    jury = Jury([make_llm(s) for s in a.judges.split(",")])
    reg = out / "regressions.jsonl"
    reg.unlink(missing_ok=True)

    traces, graphs, rows = [], [], []
    for task in ehr.tasks:
        t = agent.run(task)
        traces.append(t)
        row = {"task_id": task.id, "kind": task.kind, "success": bool(t.success), "answer": t.final_answer, "gold": task.gold}
        if not t.success:
            g = find_root_cause(agent, task, t)
            att = all_at_once(jury, t)
            row |= {"replay_root": g.root_step, "judge_step": att.step, "judge_mode": att.mode.value,
                    "judge_agent_step_kind": t.steps[att.step].kind if att.step is not None and att.step < len(t.steps) else None,
                    "root_kind": t.steps[g.root_step].kind if g.root_step is not None else None}
            graphs.append(g.to_dict() | {"dot": g.to_dot()})
            export_case(t, g, a.seed, a.patients, a.anomaly_rate, mode=att.mode.value, path=reg)
        rows.append(row)
        print(row, flush=True)

    fails = [r for r in rows if not r["success"]]
    rooted = [r for r in fails if r["replay_root"] is not None]
    agree = [r for r in rooted if r["judge_step"] == r["replay_root"]]
    summary = {
        "agent": a.agent, "judges": a.judges, "anomaly_rate": a.anomaly_rate, "n_tasks": len(rows),
        "success_rate": 1 - len(fails) / len(rows),
        "success_by_kind": {k: sum(r["success"] for r in rows if r["kind"] == k) / sum(r["kind"] == k for r in rows)
                            for k in sorted({r["kind"] for r in rows})},
        "n_failures": len(fails), "failures_root_caused_by_replay": len(rooted),
        "judge_agrees_with_replay_step": len(agree),
        "replay_root_kind": dict(Counter(r["root_kind"] for r in rooted)),
        "judge_mode": dict(Counter(r["judge_mode"] for r in fails)),
    }
    out.mkdir(parents=True, exist_ok=True)
    save_traces(traces, out / "traces.jsonl")
    (out / "graphs.json").write_text(json.dumps(graphs, indent=1))
    (out / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
