"""Failure attribution: which agent failed, and at which step (the 'first wrong span')."""
from __future__ import annotations

from dataclasses import dataclass

from .judge import Jury, judge_step, majority
from .taxonomy import FailureMode, parse_mode, taxonomy_prompt
from .trace import Trace


@dataclass
class Attribution:
    agent: str | None
    step: int | None
    mode: FailureMode = FailureMode.OTHER
    rationale: str = ""
    method: str = ""


ALL_AT_ONCE_SYSTEM = (
    "You diagnose why a multi-step AI agent run FAILED. Identify the agent responsible for the decisive error "
    "and the index of the step where that error first occurred (the first step whose output is wrong given its "
    "input). Use the step numbers shown.\nFailure taxonomy:\n" + taxonomy_prompt() + "\n"
    'Reply with JSON only: {"agent": "<name>", "step": <int>, "mode": "<taxonomy id>", "reason": "<one sentence>"}'
)


def _agent_steps(trace: Trace) -> list[int]:
    return [s.idx for s in trace.steps if s.kind != "input"]


def all_at_once(jury: Jury, trace: Trace, context: str = "") -> Attribution:
    user = (f"Task: {trace.task}\nGround truth answer: {trace.gold}\n"
            f"{('Note: ' + context) if context else ''}\n\nTrace:\n{trace.render()}\n\nJSON:")
    votes = jury.ask(ALL_AT_ONCE_SYSTEM, user)
    agent, _ = majority([v.get("agent") for v in votes])
    steps = [int(v["step"]) for v in votes if str(v.get("step", "")).lstrip("-").isdigit()]
    step, _ = majority(steps or [None])
    mode, _ = majority([parse_mode(v.get("mode")) for v in votes if v.get("mode")] or [None])
    reason = next((v.get("reason", "") for v in votes if v.get("reason")), "")
    return Attribution(agent, step, mode or FailureMode.OTHER, reason, "all_at_once")


def step_by_step(jury: Jury, trace: Trace, context: str = "", max_steps: int = 40) -> Attribution:
    """Walk the trace in order and stop at the first step the jury marks wrong."""
    for idx in _agent_steps(trace)[:max_steps]:
        v = judge_step(jury, trace, idx, context)
        if v.is_error:
            return Attribution(trace.steps[idx].agent, idx, v.mode, v.rationale, "step_by_step")
    return Attribution(None, None, method="step_by_step")
