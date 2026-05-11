"""State types for the LangGraph pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from aegis.core.constants import ContentType
from aegis.core.models import AgentFinding


class AegisState(TypedDict):
    """State for the Aegis analysis pipeline."""

    # Input
    content: str
    content_type: ContentType
    session_id: str | None
    metadata: dict[str, Any]

    # Extraction results
    extracted_text: str
    processed_data: dict[str, Any]

    # Guardrails
    guardrails_result: dict[str, Any]
    sanitized_content: str
    should_block: bool

    # Agent findings
    structural_finding: AgentFinding | None
    semantic_finding: AgentFinding | None
    intent_finding: AgentFinding | None
    visual_finding: AgentFinding | None
    behavioral_finding: AgentFinding | None

    # Collection of all findings
    findings: Annotated[list[AgentFinding], operator.add]

    # Verdict
    verdict_finding: AgentFinding | None
    risk_score: float
    risk_level: str
    is_injection: bool
    confidence: float
    summary: str

    # Pipeline metadata
    job_id: str
    processing_time_ms: int
    error: str | None
