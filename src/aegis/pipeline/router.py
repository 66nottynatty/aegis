"""Routing logic for the LangGraph pipeline."""

from __future__ import annotations

from aegis.core.constants import ContentType
from aegis.pipeline.state import AegisState


def should_block_content(state: AegisState) -> str:
    """Determine if content should be blocked early."""
    if state.get("should_block", False):
        return "block"
    return "continue"


def should_analyze_deep(state: AegisState) -> str:
    """Determine if deep analysis is needed based on content type and initial signals."""
    content_type = state["content_type"]

    # Visual analysis only for non-text content
    if content_type in (ContentType.IMAGE, ContentType.PDF):
        return "full"

    # For HTML, always do structural
    if content_type == ContentType.HTML:
        return "full"

    # For text, check if initial signals warrant deep analysis
    guardrails = state.get("guardrails_result", {})
    if guardrails.get("blocked"):
        return "full"

    # Check semantic score for early signal
    semantic = state.get("semantic_finding")
    if semantic and semantic.score > 0.3:
        return "full"

    # Check structural score for early signal
    structural = state.get("structural_finding")
    if structural and structural.score > 0.3:
        return "full"

    # Check visual score for early signal
    visual = state.get("visual_finding")
    if visual and visual.score > 0.3:
        return "full"

    # Default to standard analysis
    return "standard"


def route_by_content_type(state: AegisState) -> list[str]:
    """Return the list of nodes to execute based on content type."""
    content_type = state["content_type"]

    # All content types get these
    base_nodes = ["semantic", "intent", "behavioral"]

    if content_type == ContentType.HTML:
        return ["structural"] + base_nodes
    elif content_type == ContentType.PDF:
        return ["structural", "visual"] + base_nodes
    elif content_type == ContentType.IMAGE:
        return ["visual"] + base_nodes
    else:  # TEXT
        return base_nodes
