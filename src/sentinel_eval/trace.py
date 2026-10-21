"""Structured agent traces (one JSON object per run, OpenTelemetry-style span list)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

StepKind = Literal["input", "thought", "tool_call", "tool_result", "message", "final"]


class Step(BaseModel):
    idx: int
    agent: str = "agent"
    kind: StepKind = "message"
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    task_id: str
    task: str
    steps: list[Step] = Field(default_factory=list)
    final_answer: str | None = None
    success: bool | None = None
    gold: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def add(self, kind: StepKind, content: str, agent: str = "agent", **meta) -> Step:
        s = Step(idx=len(self.steps), agent=agent, kind=kind, content=content, meta=meta)
        self.steps.append(s)
        return s

    def render(self, upto: int | None = None, max_chars: int = 1500) -> str:
        out = []
        for s in self.steps[: None if upto is None else upto + 1]:
            c = s.content if len(s.content) <= max_chars else s.content[:max_chars] + " ...[truncated]"
            out.append(f"[step {s.idx}] ({s.agent}/{s.kind}) {c}")
        return "\n".join(out)


def save_traces(traces: list[Trace], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(t.model_dump_json() + "\n")


def load_traces(path: str | Path) -> list[Trace]:
    with open(path, encoding="utf-8") as f:
        return [Trace.model_validate_json(line) for line in f if line.strip()]
