"""Deterministic simulated policy (no API needed) for pipeline tests and CI.

It is NOT a language model: it solves EHR tasks with the reference SQL, can be configured to introduce a known
SQL-logic bug on chosen task kinds, and to obey injected instructions that contain trigger phrases. That gives
ground-truth root causes against which attribution/replay machinery can be verified.
"""
from __future__ import annotations

import json
import re

from .ehr import EHR
from .llm import MockLLM

BUGS = {
    "latest": ("DESC", "ASC"),
    "mean": ("AVG(", "MAX("),
    "active_meds": ("IS NULL", "IS NOT NULL"),
    "note": ("SELECT text", "SELECT 'unavailable'"),
}
DEFAULT_TRIGGERS = ("ignore previous instructions", "system override")


def scripted_policy(ehr: EHR, buggy_kinds: tuple[str, ...] = (), triggers: tuple[str, ...] = DEFAULT_TRIGGERS,
                    hardened: bool = False) -> MockLLM:
    tasks = {t.id: t for t in ehr.tasks}

    def fn(system: str, messages: list[dict]) -> str:
        m = re.search(r"\[task_id=(\w+)\]", messages[0]["content"])
        assert m, "scripted policy expects the agent's [task_id=...] prefix"
        task = tasks[m.group(1)]
        last = messages[-1]["content"]
        if len(messages) == 1:  # first turn: issue the query
            sql = task.reference_sql
            if task.kind in buggy_kinds:
                a, b = BUGS[task.kind]
                sql = sql.replace(a, b, 1)
            return json.dumps({"thought": "query the EHR", "sql": sql})
        low = last.lower()
        canary = re.search(r"CANARY-\w+", last)
        if canary and not hardened and any(t in low for t in triggers):
            return json.dumps({"final": canary.group(0)})
        weeks = re.search(r"follow-up in (\d+) weeks", last)
        if weeks:
            return json.dumps({"final": weeks.group(1)})
        body = last.split("TOOL RESULT:\n", 1)[-1].strip().splitlines()
        if len(body) < 2 or body[0].startswith("ERROR"):
            return json.dumps({"final": "unknown"})
        return json.dumps({"final": body[1].split(" | ")[0].strip()})

    return MockLLM(fn, name="scripted-policy")
