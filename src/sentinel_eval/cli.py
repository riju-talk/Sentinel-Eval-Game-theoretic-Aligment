"""`sentinel-eval` command line."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import whowhen
from .agent import ReActAgent
from .attacks import LLMAttacker, TemplateAttacker, evaluate_attacks
from .attribution import all_at_once, step_by_step
from .ehr import build_ehr
from .judge import Jury
from .llm import make_llm
from .regression import DEFAULT_PATH, export_case, run_regressions
from .replay import find_root_cause
from .sim import scripted_policy
from .trace import save_traces


def _sim_agent_factory(buggy=(), hardened=False):
    return lambda ehr: ReActAgent(scripted_policy(ehr, tuple(buggy), hardened=hardened), ehr)


def _load_factory(spec: str | None, buggy=(), hardened=False):
    """`module:function` returning `make_agent(ehr) -> ReActAgent`; default is the scripted simulator."""
    if not spec:
        return _sim_agent_factory(buggy, hardened)
    mod, _, fn = spec.partition(":")
    sys.path.insert(0, ".")
    return getattr(importlib.import_module(mod), fn)


def cmd_download(a):
    whowhen.download()
    print("Who&When ready in", whowhen.DEFAULT_DIR)


def cmd_whowhen(a):
    traces = whowhen.load(a.subset)[: a.limit]
    if a.method == "random":
        res = whowhen.evaluate(traces, whowhen.random_attributor(a.seed))
        res["expected_random"] = whowhen.expected_random(traces)
    else:
        jury = Jury([make_llm(s) for s in a.judges.split(",")])
        fn = all_at_once if a.method == "all" else step_by_step
        res = whowhen.evaluate(traces, lambda t: fn(jury, t))
    out = Path(a.out or f"runs/whowhen_{a.subset}_{a.method}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.subset}/{a.method}: n={res['n']} agent_acc={res['agent_acc']:.3f} step_acc={res['step_acc']:.3f} -> {out}")


def cmd_ensemble(a):
    runs = [json.loads(Path(p).read_text()) for p in a.runs]
    res = whowhen.ensemble(runs, whowhen.load(a.subset))
    print(f"jury of {res['judges']}: n={res['n']} agent_acc={res['agent_acc']:.3f} step_acc={res['step_acc']:.3f}")
    for k in ("agent_ok", "step_ok"):  # 95% bootstrap CI of the first judge, as a guide to the noise at this n
        lo, hi = whowhen.bootstrap_ci([p[k] for p in runs[0]["per_task"]])
        print(f"  judge-1 {k[:-3]} 95% CI [{lo:.3f}, {hi:.3f}]")
    for p, r in zip(a.runs, runs):
        print(f"  {Path(p).name}: agent={r['agent_acc']:.3f} step={r['step_acc']:.3f}")


def cmd_demo(a):
    """End-to-end, offline: buggy agent -> traces -> counterfactual root cause -> graphs -> regression cases."""
    out = Path(a.out)
    ehr = build_ehr(a.patients, a.seed, a.anomaly_rate)
    agent = _sim_agent_factory(buggy=a.bug)(ehr)
    traces, graphs = [], []
    reg_path = out / "regressions.jsonl"
    reg_path.unlink(missing_ok=True)
    for task in ehr.tasks:
        t = agent.run(task)
        traces.append(t)
        if not t.success:
            g = find_root_cause(agent, task, t)
            graphs.append(g.to_dict() | {"dot": g.to_dot()})
            export_case(t, g, a.seed, a.patients, a.anomaly_rate, path=reg_path)
    save_traces(traces, out / "traces.jsonl")
    (out / "graphs.json").write_text(json.dumps(graphs, indent=1))
    ok = sum(bool(t.success) for t in traces)
    rooted = sum(g["root_step"] is not None for g in graphs)
    print(f"tasks={len(traces)} success={ok} failures={len(graphs)} root-caused-by-replay={rooted} -> {out}")


def cmd_attack(a):
    factory = _load_factory(a.agent, hardened=a.hardened)
    atk = LLMAttacker(make_llm(a.attacker)) if a.attacker else TemplateAttacker(a.seed)
    r = evaluate_attacks(factory, atk, seed=a.seed)
    print(f"n={r.n} ASR={r.asr:.2f} task_success_attacked={r.task_success_attacked:.2f} "
          f"task_success_clean={r.task_success_clean:.2f}")


def cmd_regress(a):
    res = run_regressions(_load_factory(a.agent, buggy=a.bug), Path(a.path))
    failed = [r for r in res if not r["passed"]]
    print(f"{len(res) - len(failed)}/{len(res)} regression cases pass")
    for r in failed:
        print("  REGRESSED", r["task_id"], "->", r["answer"], "(gold", r["gold"] + ")")
    sys.exit(1 if failed else 0)


def main(argv=None):
    p = argparse.ArgumentParser("sentinel-eval")
    sub = p.add_subparsers(required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)

    w = sub.add_parser("whowhen", help="failure-attribution accuracy on Who&When")
    w.add_argument("--subset", choices=["hand", "algo"], default="algo")
    w.add_argument("--method", choices=["random", "all", "step"], default="random")
    w.add_argument("--judges", default="anthropic:claude-sonnet-5-5", help="comma-separated LLM specs (the jury)")
    w.add_argument("--limit", type=int, default=None)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--out")
    w.set_defaults(fn=cmd_whowhen)

    e = sub.add_parser("ensemble", help="majority-vote jury over per-judge whowhen result files")
    e.add_argument("runs", nargs="+")
    e.add_argument("--subset", choices=["hand", "algo"], default="algo")
    e.set_defaults(fn=cmd_ensemble)

    d = sub.add_parser("demo", help="offline end-to-end demo with the simulated agent")
    d.add_argument("--bug", nargs="*", default=["latest", "mean"])
    d.add_argument("--patients", type=int, default=12)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--anomaly-rate", type=float, default=0.0)
    d.add_argument("--out", default="runs/demo")
    d.set_defaults(fn=cmd_demo)

    t = sub.add_parser("attack", help="prompt-injection attack success vs task success")
    t.add_argument("--agent", help="module:factory (default: simulator)")
    t.add_argument("--attacker", help="LLM spec for an LLM attacker (default: templates)")
    t.add_argument("--hardened", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_attack)

    r = sub.add_parser("regress", help="replay recorded failures; exit 1 on regression")
    r.add_argument("--agent")
    r.add_argument("--bug", nargs="*", default=[])
    r.add_argument("--path", default=str(DEFAULT_PATH))
    r.set_defaults(fn=cmd_regress)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
