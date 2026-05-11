"""Pipeline nodes for the LangGraph analysis flow."""

from __future__ import annotations

import logging
from typing import Any

from aegis.agents.behavioral import BehavioralAgent
from aegis.agents.intent import IntentAgent
from aegis.agents.memory_agent import MemoryAgent
from aegis.agents.semantic import SemanticAgent
from aegis.agents.structural import StructuralAgent
from aegis.agents.verdict import VerdictAgent
from aegis.agents.visual import VisualAgent
from aegis.core.constants import ContentType
from aegis.core.models import AgentFinding, ScanResult
from aegis.guardrails.rails import check_guardrails
from aegis.pipeline.state import AegisState
from aegis.processors.html import HtmlProcessor
from aegis.processors.image import ImageProcessor
from aegis.processors.pdf import PdfProcessor
from aegis.processors.text import TextProcessor

logger = logging.getLogger(__name__)

# Initialize processors
_html_processor = HtmlProcessor()
_pdf_processor = PdfProcessor()
_image_processor = ImageProcessor()
_text_processor = TextProcessor()

# Initialize agents
_structural_agent = StructuralAgent(enable_memory=True)
_semantic_agent = SemanticAgent(enable_memory=True)
_intent_agent = IntentAgent(enable_memory=True)
_visual_agent = VisualAgent(enable_memory=True)
_behavioral_agent = BehavioralAgent(enable_memory=True)
_verdict_agent = VerdictAgent(enable_memory=True)
_memory_agent = MemoryAgent(enable_memory=True)


def extract_node(state: AegisState) -> dict[str, Any]:
    """Extract and process content based on type."""
    content = state["content"]
    content_type = state["content_type"]

    try:
        if content_type == ContentType.HTML:
            processed = _html_processor.process(content)
            return {
                "extracted_text": processed.all_text or content,
                "processed_data": processed.to_dict(),
            }
        elif content_type == ContentType.PDF:
            processed = _pdf_processor.process(content)
            return {
                "extracted_text": processed.text or "",
                "processed_data": processed.to_dict(),
            }
        elif content_type == ContentType.IMAGE:
            processed = _image_processor.process(content)
            return {
                "extracted_text": processed.ocr_text or "",
                "processed_data": processed.to_dict(),
            }
        else:  # TEXT
            processed = _text_processor.process(content)
            return {
                "extracted_text": processed.normalized,
                "processed_data": processed.to_dict(),
            }
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return {
            "extracted_text": content,
            "processed_data": {"error": str(exc)},
            "error": f"Extraction failed: {exc}",
        }


def sanitize_node(state: AegisState) -> dict[str, Any]:
    """Sanitize content by removing obvious injection attempts."""
    text = state.get("extracted_text", "")

    # Basic sanitization: truncate very long inputs
    max_len = 100000
    if len(text) > max_len:
        text = text[:max_len] + "... [truncated]"

    return {
        "sanitized_content": text,
    }


def guardrails_node(state: AegisState) -> dict[str, Any]:
    """Run NeMo+Ollama guardrails check."""
    text = state.get("sanitized_content", "")

    try:
        result = check_guardrails(text)
        should_block = result.get("blocked", False)
        
        findings = []
        if should_block:
            finding = AgentFinding(
                agent="guardrails",
                score=result.get("score", 1.0),
                signals=result.get("patterns_matched", ["guardrails_block"]),
                explanation=result.get("reason", "Content blocked by guardrails"),
                metadata={"source": result.get("source", "unknown")}
            )
            findings.append(finding)
            return {
                "guardrails_result": result,
                "should_block": True,
                "findings": findings,
                "risk_score": result.get("score", 1.0),
                "risk_level": "high" if result.get("score", 0.0) > 0.7 else "medium",
            }

        return {
            "guardrails_result": result,
            "should_block": False,
        }
    except Exception as exc:
        logger.warning("Guardrails check failed: %s", exc)
        return {
            "guardrails_result": {"blocked": False, "error": str(exc)},
            "should_block": False,
        }


def structural_node(state: AegisState) -> dict[str, Any]:
    """Run structural analysis."""
    content = state["content"]
    context = {
        "content_type": state["content_type"].value,
        "session_id": state.get("session_id"),
        "metadata": state.get("metadata", {}),
        "processed": state.get("processed_data", {}),
    }

    try:
        finding = _structural_agent.analyze(content, context)
        return {
            "structural_finding": finding,
            "findings": [finding],
        }
    except Exception as exc:
        logger.error("Structural agent failed: %s", exc)
        error_finding = AgentFinding(
            agent="structural",
            score=0.0,
            signals=["agent_error"],
            explanation=f"Structural agent failed: {exc}",
        )
        return {
            "structural_finding": error_finding,
            "findings": [error_finding],
        }


def semantic_node(state: AegisState) -> dict[str, Any]:
    """Run semantic analysis."""
    content = state.get("sanitized_content", state["content"])
    context = {
        "content_type": state["content_type"].value,
        "session_id": state.get("session_id"),
        "metadata": state.get("metadata", {}),
        "processed": state.get("processed_data", {}),
    }

    try:
        finding = _semantic_agent.analyze(content, context)
        return {
            "semantic_finding": finding,
            "findings": [finding],
        }
    except Exception as exc:
        logger.error("Semantic agent failed: %s", exc)
        error_finding = AgentFinding(
            agent="semantic",
            score=0.0,
            signals=["agent_error"],
            explanation=f"Semantic agent failed: {exc}",
        )
        return {
            "semantic_finding": error_finding,
            "findings": [error_finding],
        }


def intent_node(state: AegisState) -> dict[str, Any]:
    """Run intent analysis."""
    content = state.get("sanitized_content", state["content"])
    context = {
        "content_type": state["content_type"].value,
        "session_id": state.get("session_id"),
        "metadata": state.get("metadata", {}),
        "processed": state.get("processed_data", {}),
    }

    try:
        finding = _intent_agent.analyze(content, context)
        return {
            "intent_finding": finding,
            "findings": [finding],
        }
    except Exception as exc:
        logger.error("Intent agent failed: %s", exc)
        error_finding = AgentFinding(
            agent="intent",
            score=0.0,
            signals=["agent_error"],
            explanation=f"Intent agent failed: {exc}",
        )
        return {
            "intent_finding": error_finding,
            "findings": [error_finding],
        }


def visual_node(state: AegisState) -> dict[str, Any]:
    """Run visual analysis."""
    content = state["content"]
    context = {
        "content_type": state["content_type"].value,
        "session_id": state.get("session_id"),
        "metadata": state.get("metadata", {}),
        "processed": state.get("processed_data", {}),
    }

    # Skip visual for text-only content
    if state["content_type"] == ContentType.TEXT:
        no_op_finding = AgentFinding(
            agent="visual",
            score=0.0,
            signals=[],
            explanation="Visual analysis not applicable for text content.",
        )
        return {
            "visual_finding": no_op_finding,
            "findings": [no_op_finding],
        }

    try:
        finding = _visual_agent.analyze(content, context)
        return {
            "visual_finding": finding,
            "findings": [finding],
        }
    except Exception as exc:
        logger.error("Visual agent failed: %s", exc)
        error_finding = AgentFinding(
            agent="visual",
            score=0.0,
            signals=["agent_error"],
            explanation=f"Visual agent failed: {exc}",
        )
        return {
            "visual_finding": error_finding,
            "findings": [error_finding],
        }


def behavioral_node(state: AegisState) -> dict[str, Any]:
    """Run behavioral analysis."""
    content = state.get("sanitized_content", state["content"])
    context = {
        "content_type": state["content_type"].value,
        "session_id": state.get("session_id"),
        "metadata": state.get("metadata", {}),
        "processed": state.get("processed_data", {}),
    }

    try:
        finding = _behavioral_agent.analyze(content, context)
        return {
            "behavioral_finding": finding,
            "findings": [finding],
        }
    except Exception as exc:
        logger.error("Behavioral agent failed: %s", exc)
        error_finding = AgentFinding(
            agent="behavioral",
            score=0.0,
            signals=["agent_error"],
            explanation=f"Behavioral agent failed: {exc}",
        )
        return {
            "behavioral_finding": error_finding,
            "findings": [error_finding],
        }


def verdict_node(state: AegisState) -> dict[str, Any]:
    """Aggregate findings and produce final verdict."""
    # Use accumulated findings from state
    all_findings = state.get("findings", [])

    context: dict[str, Any] = {
        "agent_findings": all_findings,
        "content_type": state["content_type"].value,
        "session_id": state.get("session_id"),
        "metadata": state.get("metadata", {}),
    }

    try:
        finding = _verdict_agent.analyze(state["content"], context)

        meta = finding.metadata
        return {
            "verdict_finding": finding,
            "findings": [finding],
            "risk_score": finding.score,
            "risk_level": meta.get("risk_level", "safe"),
            "is_injection": meta.get("is_injection", False),
            "confidence": meta.get("confidence", 0.5),
            "summary": meta.get("summary", ""),
        }
    except Exception as exc:
        logger.error("Verdict agent failed: %s", exc)
        return {
            "verdict_finding": None,
            "risk_score": 0.0,
            "risk_level": "safe",
            "is_injection": False,
            "confidence": 0.0,
            "summary": f"Verdict failed: {exc}",
        }


def memory_node(state: AegisState) -> dict[str, Any]:
    """Persist findings to memory."""
    from aegis.core.constants import RiskLevel

    # Build a scan result for memory storage
    scan_result = ScanResult(
        job_id=state.get("job_id", "unknown"),
        risk_level=RiskLevel(state.get("risk_level", "safe")),
        risk_score=state.get("risk_score", 0.0),
        is_injection=state.get("is_injection", False),
        confidence=state.get("confidence", 0.0),
        findings=state.get("findings", []),
        summary=state.get("summary", ""),
        content_type=state["content_type"],
        processing_time_ms=state.get("processing_time_ms", 0),
        sanitized_content=state.get("sanitized_content"),
    )

    context: dict[str, Any] = {
        "scan_result": scan_result,
        "session_id": state.get("session_id"),
    }

    try:
        _memory_agent.analyze(state["content"], context)
    except Exception as exc:
        logger.debug("Memory storage failed: %s", exc)

    return {}
