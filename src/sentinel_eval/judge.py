"""LLM-as-judge with the bias mitigations from the report: binary verdicts, jury majority, order swapping."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

from .llm import LLM
from .taxonomy import FailureMode, parse_mode, taxonomy_prompt
from .trace import Trace


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (tolerates fences and prose). Returns {} on failure."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return {}


class Jury:
    """N judge models (or N samples of one model); verdicts are aggregated by majority vote."""

    def __init__(self, judges: list[LLM], temperature: float = 0.0):
        if not judges:
            raise ValueError("a jury needs at least one judge")
        self.judges, self.temperature = judges, temperature

    def ask(self, system: str, user: str, max_tokens: int = 700) -> list[dict]:
        votes = []
        for j in self.judges:
            try:
                msgs = [{"role": "user", "content": user}]
                reply = j.complete(system, msgs, self.temperature, max_tokens)
                vote = extract_json(reply)
                if not vote:  # small models sometimes answer in prose: retry once with a format reminder
                    msgs += [{"role": "assistant", "content": reply[:500]},
                             {"role": "user", "content": "Reply with ONLY the JSON object described, nothing else."}]
                    vote = extract_json(j.complete(system, msgs, self.temperature, max_tokens))
                votes.append(vote)
            except Exception as e:  # a failing judge abstains rather than crashing the run, but is never silent
                print(f"[jury] {j.name} failed: {type(e).__name__}: {e}", file=sys.stderr)
                votes.append({"_error": str(e)})
        return votes


def majority(values: list) -> tuple[object | None, float]:
    """Most common non-None value and its vote share among the non-None votes."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, 0.0
    v, n = Counter(vals).most_common(1)[0]
    return v, n / len(vals)


@dataclass
class StepVerdict:
    step: int
    is_error: bool
    mode: FailureMode = FailureMode.OTHER
    rationale: str = ""
    agreement: float = 1.0
    votes: list[dict] = field(default_factory=list)


STEP_SYSTEM = (
    "You audit an AI agent's execution trace. A step is WRONG if its output is incorrect, unsupported, "
    "or unhelpful GIVEN only its input and the steps before it. Judge only the step you are asked about.\n"
    "Failure taxonomy:\n" + taxonomy_prompt() + "\n"
    'Reply with JSON only: {"error": true|false, "mode": "<taxonomy id or null>", "reason": "<one sentence>"}'
)


def judge_step(jury: Jury, trace: Trace, idx: int, context: str = "") -> StepVerdict:
    user = (f"Task: {trace.task}\n{('Reference: ' + context) if context else ''}\n\n"
            f"Trace so far:\n{trace.render(idx)}\n\nIs step {idx} wrong? Answer with the JSON object.")
    votes = jury.ask(STEP_SYSTEM, user)
    flags = [bool(v["error"]) if "error" in v else None for v in votes]
    is_err, share = majority(flags)
    mode, _ = majority([parse_mode(v.get("mode")) for v in votes if v.get("error")] or [None])
    reason = next((v.get("reason", "") for v in votes if bool(v.get("error")) == is_err and v.get("reason")), "")
    return StepVerdict(idx, bool(is_err), mode or FailureMode.OTHER, reason, share, votes)


PAIR_SYSTEM = ('You compare two candidate answers to a task. Reply with JSON only: {"winner": "A"|"B"|"tie"}')


def pairwise(jury: Jury, task: str, a: str, b: str) -> str:
    """Judge A-vs-B in both presentation orders; return 'a', 'b' or 'tie' (disagreement across orders = tie)."""
    def run(first: str, second: str) -> list[str | None]:
        u = f"Task: {task}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}"
        return [v.get("winner") for v in jury.ask(PAIR_SYSTEM, u)]

    fwd, _ = majority(run(a, b))
    rev, _ = majority(run(b, a))
    rev = {"A": "B", "B": "A"}.get(rev, rev)  # map back to the original labelling
    if fwd == rev and fwd in ("A", "B"):
        return fwd.lower()
    return "tie"
