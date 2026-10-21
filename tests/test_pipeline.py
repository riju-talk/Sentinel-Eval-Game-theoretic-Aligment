import json

from sentinel_eval.agent import ReActAgent
from sentinel_eval.attacks import TemplateAttacker, evaluate_attacks
from sentinel_eval.attribution import all_at_once, step_by_step
from sentinel_eval.ehr import build_ehr, inject_anomalies
from sentinel_eval.judge import Jury, extract_json, majority, pairwise
from sentinel_eval.llm import MockLLM
from sentinel_eval.regression import export_case, run_regressions
from sentinel_eval.replay import find_root_cause
from sentinel_eval.sim import scripted_policy
from sentinel_eval.taxonomy import FailureMode


def agent_for(buggy=(), hardened=False):
    def make(ehr):
        return ReActAgent(scripted_policy(ehr, buggy, hardened=hardened), ehr)
    return make


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert extract_json("no json") == {}


def test_clean_agent_solves_all_tasks():
    ehr = build_ehr(8, seed=1)
    agent = agent_for()(ehr)
    assert all(agent.run(t).success for t in ehr.tasks)


def test_counterfactual_replay_localizes_sql_bug_to_tool_call():
    ehr = build_ehr(8, seed=1)
    agent = agent_for(buggy=("latest",))(ehr)
    task = next(t for t in ehr.tasks if t.kind == "latest")
    trace = agent.run(task)
    assert not trace.success
    g = find_root_cause(agent, task, trace)
    # the bug is in the query (step 1); patching it fixes the run, so it is the causal step
    assert g.root_step == 1 and trace.steps[1].kind == "tool_call"
    assert g.interventions[-1].fixed
    assert "digraph" in g.to_dot()


def test_replay_of_correct_trace_has_no_root():
    ehr = build_ehr(6, seed=2)
    agent = agent_for()(ehr)
    task = ehr.tasks[0]
    assert find_root_cause(agent, task, agent.run(task)).root_step is None


def test_anomalies_degrade_agent_but_gold_stays_clean():
    ehr = build_ehr(10, seed=3)
    n = inject_anomalies(ehr, 0.6, 3)
    assert n > 0
    agent = agent_for()(ehr)
    assert not all(agent.run(t).success for t in ehr.tasks if t.kind in ("latest", "mean"))


def test_prompt_injection_asr_and_hardening():
    atk = TemplateAttacker(seed=0)
    vulnerable = evaluate_attacks(agent_for(), atk)
    hardened = evaluate_attacks(agent_for(hardened=True), TemplateAttacker(seed=0))
    assert vulnerable.asr > 0
    assert hardened.asr == 0
    assert hardened.task_success_attacked == hardened.task_success_clean == 1.0  # hardened without refusing


def test_jury_majority_and_order_swap():
    yes = MockLLM(lambda s, m: '{"winner": "A"}')
    # a judge that always says "A" is pure position bias: order swapping turns it into a tie
    assert pairwise(Jury([yes]), "q", "x", "y") == "tie"
    smart = MockLLM(lambda s, m: json.dumps({"winner": "A" if m[0]["content"].index("good") < m[0]["content"].index("bad") else "B"}))
    assert pairwise(Jury([smart]), "q", "good", "bad") == "a"
    assert majority([True, True, False]) == (True, 2 / 3)


def test_attribution_with_mock_judges():
    ehr = build_ehr(6, seed=1)
    agent = agent_for(buggy=("latest",))(ehr)
    trace = agent.run(next(t for t in ehr.tasks if t.kind == "latest"))

    def judge(system, messages):
        u = messages[0]["content"]
        if "Is step" in u:
            idx = int(u.split("Is step ")[1].split()[0])
            return json.dumps({"error": idx == 1, "mode": "sql_logic_error", "reason": "wrong ordering"})
        return json.dumps({"agent": "agent", "step": 1, "mode": "sql_logic_error", "reason": "x"})

    jury = Jury([MockLLM(judge)] * 3)
    a = step_by_step(jury, trace)
    assert (a.step, a.mode) == (1, FailureMode.SQL_LOGIC_ERROR)
    assert all_at_once(jury, trace).step == 1


def test_regression_roundtrip(tmp_path):
    p = tmp_path / "cases.jsonl"
    ehr = build_ehr(8, seed=1)
    agent = agent_for(buggy=("latest",))(ehr)
    task = next(t for t in ehr.tasks if t.kind == "latest")
    trace = agent.run(task)
    export_case(trace, find_root_cause(agent, task, trace), 1, 8, path=p)
    assert not run_regressions(agent_for(buggy=("latest",)), p)[0]["passed"]   # bad version still fails
    assert run_regressions(agent_for(), p)[0]["passed"]                        # fixed version passes
