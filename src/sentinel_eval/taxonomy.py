"""Failure taxonomy: the report's seven categories plus the two procedural modes from EHR-Complex."""
from __future__ import annotations

from enum import Enum


class FailureMode(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    GROUNDING_FAILURE = "grounding_failure"
    TOOL_MISUSE = "tool_misuse"
    REASONING_ERROR = "reasoning_error"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONTEXT_HANDLING = "context_handling"
    STATE_MANAGEMENT = "state_management"
    SQL_LOGIC_ERROR = "sql_logic_error"
    CODE_LOOKUP_ERROR = "medical_code_lookup_error"
    INJECTION_COMPLIANCE = "injection_compliance"
    OTHER = "other"


DEFINITIONS: dict[FailureMode, str] = {
    FailureMode.RETRIEVAL_MISS: "The correct information existed in the environment but was not retrieved.",
    FailureMode.GROUNDING_FAILURE: "Good context was retrieved, but the output contradicted or ignored it.",
    FailureMode.TOOL_MISUSE: "Wrong tool, incorrect arguments, or a needed tool was not called.",
    FailureMode.REASONING_ERROR: "Incorrect conclusion from correct facts, or a calculation error.",
    FailureMode.INSTRUCTION_FOLLOWING: "Output violated format, length, tone, or policy instructions.",
    FailureMode.CONTEXT_HANDLING: "Context was lost, or data from different patients/sessions was mixed up.",
    FailureMode.STATE_MANAGEMENT: "Internal state was not carried correctly across steps.",
    FailureMode.SQL_LOGIC_ERROR: "Syntactically valid query with wrong logic (bad join, filter, aggregation, time window).",
    FailureMode.CODE_LOOKUP_ERROR: "Wrong or missing medical/ontology code (ICD, LOINC, SNOMED, itemid) used in a lookup.",
    FailureMode.INJECTION_COMPLIANCE: "The agent followed an instruction embedded in untrusted tool output.",
    FailureMode.OTHER: "None of the above.",
}


def taxonomy_prompt() -> str:
    return "\n".join(f"- {m.value}: {d}" for m, d in DEFINITIONS.items())


def parse_mode(s: str | None) -> FailureMode:
    try:
        return FailureMode((s or "").strip().lower())
    except ValueError:
        return FailureMode.OTHER
