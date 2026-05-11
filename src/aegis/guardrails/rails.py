"""Tiered classifier architecture for content safety."""

from __future__ import annotations

import logging
from typing import Any

from aegis.classifiers.deberta import DebertaClassifier
from aegis.classifiers.llama_guard import LlamaGuardClassifier
from aegis.classifiers.rule_based import RuleBasedClassifier

logger = logging.getLogger(__name__)

# Initialize classifiers lazily
_rule_classifier = RuleBasedClassifier()
_deberta_classifier: DebertaClassifier | None = None
_llama_guard_classifier: LlamaGuardClassifier | None = None


def _get_deberta() -> DebertaClassifier:
    global _deberta_classifier
    if _deberta_classifier is None:
        _deberta_classifier = DebertaClassifier()
    return _deberta_classifier


def _get_llama_guard() -> LlamaGuardClassifier:
    global _llama_guard_classifier
    if _llama_guard_classifier is None:
        _llama_guard_classifier = LlamaGuardClassifier()
    return _llama_guard_classifier


def check_guardrails(text: str) -> dict[str, Any]:
    """
    Check content against tiered classifiers.

    1. Tier 1: Rule-based (Regex)
    2. Tier 2: DeBERTa (ML)
    3. Tier 3: Llama Guard (LLM) - only if Tier 2 is ambiguous

    Args:
        text: The content to check

    Returns:
        Dictionary with decision and metadata
    """
    # --- Tier 1: Rule-based ---
    rule_result = _rule_classifier.predict(text)
    if rule_result.score >= 0.7:
        return {
            "blocked": True,
            "reason": f"Tier 1: Rule-based match - {rule_result.matched_patterns[:2]}",
            "score": rule_result.score,
            "patterns_matched": rule_result.matched_patterns,
            "source": "rule_based",
        }

    # --- Tier 2: DeBERTa ---
    deberta = _get_deberta()
    if deberta.is_available():
        deberta_result = deberta.predict(text)
        
        if deberta_result.score > 0.7:
            return {
                "blocked": True,
                "reason": "Tier 2: DeBERTa ML classifier high-risk detection",
                "score": deberta_result.score,
                "patterns_matched": deberta_result.matched_patterns,
                "source": "deberta",
            }
        
        if deberta_result.score < 0.35:
            return {
                "blocked": False,
                "reason": None,
                "score": deberta_result.score,
                "patterns_matched": [],
                "source": "deberta",
            }
        
        # Ambiguous (0.35 - 0.70) -> Fall through to Tier 3
        logger.info("DeBERTa score ambiguous (%.3f), falling back to Tier 3", deberta_result.score)
    else:
        logger.warning("DeBERTa classifier not available, falling back to Tier 3")

    # --- Tier 3: Llama Guard ---
    llama_guard = _get_llama_guard()
    if llama_guard.is_available():
        lg_result = llama_guard.predict(text)
        return {
            "blocked": lg_result.label == "injection",
            "reason": f"Tier 3: Llama Guard analysis: {lg_result.matched_patterns[0] if lg_result.matched_patterns else 'Unknown'}",
            "score": lg_result.score,
            "patterns_matched": lg_result.matched_patterns,
            "source": "llama_guard",
        }

    # Default fallback if LLM is unavailable
    return {
        "blocked": False,
        "reason": "Classifiers completed with no high-risk detection",
        "score": 0.0,
        "patterns_matched": [],
        "source": "default",
    }


def sanitize_content(text: str) -> str:
    """
    Sanitize content by removing/replacing suspicious patterns.

    Args:
        text: Content to sanitize

    Returns:
        Sanitized content
    """
    # Remove null bytes
    sanitized = text.replace("\x00", "")

    # Remove zero-width characters
    from aegis.core.constants import ZERO_WIDTH_CHARS
    for char in ZERO_WIDTH_CHARS:
        sanitized = sanitized.replace(char, "")

    # Normalize excessive whitespace
    import re
    sanitized = re.sub(r"\s+", " ", sanitized)

    # Truncate if extremely long
    max_len = 100000
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "\n[Content truncated due to length]"

    return sanitized.strip()
