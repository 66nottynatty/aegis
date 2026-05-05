"""Semantic Agent - Three-tier prompt injection detection system.

Implements a cascade of classifiers:
- Tier 1: Rule-based regex (always runs, catches obvious cases)
- Tier 2: DeBERTa model (runs if Tier 1 doesn't find high-confidence injection)
- Tier 3: Llama Guard 3 via Ollama (runs only if Tier 2 result is ambiguous)

This tiered approach provides both speed and accuracy, handling the full
spectrum from obvious injections to subtle adversarial patterns.
"""

from __future__ import annotations

import logging
from typing import Any

from aegis.agents.base import BaseAegisAgent
from aegis.classifiers import (
    ClassifierResult,
    DebertaClassifier,
    LlamaGuardClassifier,
    RuleBasedClassifier,
)
from aegis.classifiers.base import BaseClassifier
from aegis.core.config import ClassifierBackend, get_config
from aegis.core.constants import AgentName
from aegis.core.models import AgentFinding

logger = logging.getLogger(__name__)


class SemanticAgent(BaseAegisAgent):
    name = AgentName.SEMANTIC
    role = "Semantic Content Classifier"
    goal = "Identify prompt injection patterns through tiered semantic analysis"
    backstory = (
        "Advanced NLP specialist using a three-tier detection pipeline. "
        "Starts with fast rule-based pattern matching, escalates to ML-based "
        "DeBERTa classification for uncertain cases, and uses LLM-based Llama Guard "
        "for final resolution of ambiguous inputs. Balances speed and accuracy "
        "to detect both obvious and subtle adversarial patterns."
    )

    HIGH_CONFIDENCE_THRESHOLD = 0.70

    def __init__(self, enable_memory: bool = True) -> None:
        super().__init__(enable_memory=enable_memory)
        self._tier1: RuleBasedClassifier | None = None
        self._tier2: DebertaClassifier | None = None
        self._tier3: LlamaGuardClassifier | None = None
        self._backend: ClassifierBackend = ClassifierBackend.TIERED
        self._init_classifiers()

    def _init_classifiers(self) -> None:
        """Initialize classifiers based on configuration."""
        config = get_config()
        self._backend = config.classifier.backend

        self._tier1 = RuleBasedClassifier()
        logger.info("Tier 1 (Rule-based) classifier initialized")

        if self._backend == ClassifierBackend.TIERED:
            try:
                self._tier2 = DebertaClassifier(device=config.classifier.device)
                if self._tier2.is_available():
                    logger.info("Tier 2 (DeBERTa) classifier initialized")
                else:
                    logger.warning("DeBERTa model unavailable, using rule-based only")
                    self._tier2 = None
            except Exception as exc:
                logger.warning("Failed to load DeBERTa: %s", exc)
                self._tier2 = None

            try:
                self._tier3 = LlamaGuardClassifier()
                if self._tier3.is_available():
                    logger.info("Tier 3 (Llama Guard) classifier initialized")
                else:
                    logger.warning("Llama Guard unavailable")
                    self._tier3 = None
            except Exception as exc:
                logger.warning("Failed to load Llama Guard: %s", exc)
                self._tier3 = None

        elif self._backend == ClassifierBackend.DEBERTA:
            try:
                self._tier2 = DebertaClassifier(device=config.classifier.device)
                if not self._tier2.is_available():
                    self._tier2 = None
            except Exception as exc:
                logger.warning("DeBERTa unavailable: %s", exc)
                self._tier2 = None

    def analyze(self, content: str, context: dict[str, Any]) -> AgentFinding:
        text = self._extract_text(content, context)
        signals: list[str] = []
        tier_scores: dict[str, float] = {}
        tier_labels: dict[str, str] = {}

        if self._backend == ClassifierBackend.TIERED:
            result, tier_info = self._tiered_analysis(text)
            signals.extend(tier_info.get("signals", []))
            tier_scores.update(tier_info.get("scores", {}))
            tier_labels.update(tier_info.get("labels", {}))

            combined_score = self._aggregate_scores(tier_scores)
            explanation = self._build_explanation(tier_scores, tier_labels, signals)
            metadata = {
                "tier_scores": tier_scores,
                "tier_labels": tier_labels,
                "backend": "tiered",
                "tier_info": tier_info,
            }

        elif self._backend == ClassifierBackend.DEBERTA:
            result = self._run_deberta_analysis(text, signals)
            combined_score = result.score
            explanation = self._build_explanation(
                {"deberta": result.score}, {"deberta": result.label}, signals
            )
            metadata = {
                "classifier_label": result.label,
                "confidence": result.confidence,
                "backend": "deberta",
            }

        else:
            result = self._fallback_analysis(text, signals)
            combined_score = result.score
            explanation = self._build_explanation(
                {"fallback": result.score}, {"fallback": result.label}, signals
            )
            metadata = {
                "classifier_label": result.label,
                "confidence": result.confidence,
                "backend": self._backend.value,
            }

        additional_signals, additional_score = self._check_additional_patterns(text)
        signals.extend(additional_signals)
        combined_score = min(1.0, combined_score + additional_score * 0.3)

        memory_results = self.search_memory(text[:200])
        if memory_results:
            signals.append("known_injection_pattern_in_memory")
            combined_score = min(1.0, combined_score + 0.1)

        if combined_score > 0.5 and signals:
            self.add_memory(
                f"Semantic injection detected: {', '.join(signals[:3])}",
                metadata={"score": combined_score},
            )

        return self._make_finding(combined_score, signals, explanation, metadata)

    def _tiered_analysis(
        self, text: str
    ) -> tuple[ClassifierResult, dict[str, Any]]:
        """Execute the three-tier analysis pipeline.

        Args:
            text: Text content to analyze

        Returns:
            Tuple of (final result, tier information dictionary)
        """
        tier_info: dict[str, Any] = {
            "tier1_triggered": False,
            "tier2_triggered": False,
            "tier3_triggered": False,
            "signals": [],
            "scores": {},
            "labels": {},
        }

        tier1_result = self._tier1.predict(text)
        tier_info["scores"]["tier1"] = tier1_result.score
        tier_info["labels"]["tier1"] = tier1_result.label

        if tier1_result.label == "injection" and tier1_result.score >= self.HIGH_CONFIDENCE_THRESHOLD:
            tier_info["tier1_triggered"] = True
            tier_info["signals"].extend(
                [f"pattern:{p[:50]}" for p in tier1_result.matched_patterns[:10]]
            )
            logger.debug("Tier 1 high-confidence detection: %.3f", tier1_result.score)
            return tier1_result, tier_info

        tier_info["signals"].extend(
            [f"tier1:{p[:30]}" for p in tier1_result.matched_patterns[:5]]
        )

        if self._tier2 is not None and self._tier2.is_available():
            tier2_result = self._tier2.predict(text)
            tier_info["scores"]["tier2"] = tier2_result.score
            tier_info["labels"]["tier2"] = tier2_result.label

            if tier2_result.label == "injection" and tier2_result.score >= self.HIGH_CONFIDENCE_THRESHOLD:
                tier_info["tier2_triggered"] = True
                tier_info["signals"].extend(tier2_result.matched_patterns)
                logger.debug("Tier 2 high-confidence detection: %.3f", tier2_result.score)
                return tier2_result, tier_info

            if not self._tier2.is_ambiguous(tier2_result.score):
                if tier2_result.label == "injection":
                    tier_info["signals"].extend(tier2_result.matched_patterns)
                logger.debug("Tier 2 conclusive result: %.3f", tier2_result.score)
                return tier2_result, tier_info

            tier_info["signals"].append(f"tier2_ambiguous:{tier2_result.score:.3f}")

            if self._tier3 is not None and self._tier3.is_available():
                tier3_result = self._tier3.predict(text)
                tier_info["scores"]["tier3"] = tier3_result.score
                tier_info["labels"]["tier3"] = tier3_result.label
                tier_info["tier3_triggered"] = True

                if tier3_result.label == "injection":
                    tier_info["signals"].extend(tier3_result.matched_patterns)

                logger.debug("Tier 3 result: %.3f", tier3_result.score)
                return tier3_result, tier_info

            logger.debug("Tier 2 ambiguous, Tier 3 unavailable - using Tier 2")
            return tier2_result, tier_info

        logger.debug("Tier 1 result only (Tier 2 unavailable)")
        return tier1_result, tier_info

    def _run_deberta_analysis(self, text: str, signals: list[str]) -> ClassifierResult:
        """Run DeBERTa analysis for DEBERTA backend mode."""
        if self._tier2 is not None and self._tier2.is_available():
            result = self._tier2.predict(text)
            if result.matched_patterns:
                signals.extend(result.matched_patterns)
            return result

        signals.append("deberta_unavailable_fallback")
        return self._tier1.predict(text)

    def _fallback_analysis(self, text: str, signals: list[str]) -> ClassifierResult:
        """Run fallback analysis for non-tiered backends."""
        signals.append(f"using_backend:{self._backend.value}")
        return self._tier1.predict(text)

    def _aggregate_scores(self, tier_scores: dict[str, float]) -> float:
        """Aggregate scores from multiple tiers.

        Uses weighted average giving more weight to higher tiers.

        Args:
            tier_scores: Dictionary of tier name to score

        Returns:
            Combined score
        """
        if not tier_scores:
            return 0.0

        weights = {"tier1": 0.3, "tier2": 0.4, "tier3": 0.3}
        total_weight = sum(weights.get(k, 0.25) for k in tier_scores)
        weighted_sum = sum(
            tier_scores[k] * weights.get(k, 0.25) for k in tier_scores
        )

        return min(1.0, weighted_sum / total_weight if total_weight else 0.0)

    def _extract_text(self, content: str, context: dict[str, Any]) -> str:
        """Extract text from content and context."""
        processed = context.get("processed", {})
        if not isinstance(processed, dict):
            return content
        if "all_text" in processed:
            return str(processed["all_text"])
        if "normalized_text" in processed:
            return str(processed["normalized_text"])
        if "ocr_text" in processed:
            return str(processed.get("ocr_text", "")) + " " + content
        if "normalized" in processed:
            return str(processed["normalized"])
        return content

    def _check_additional_patterns(self, text: str) -> tuple[list[str], float]:
        """Check for additional obfuscation patterns."""
        import re

        signals = []
        score = 0.0
        text_lower = text.lower()

        instruction_sequences = [
            ("first", "then", "finally"),
            ("step 1", "step 2"),
            ("1.", "2.", "3."),
        ]
        for seq in instruction_sequences:
            if all(s in text_lower for s in seq[:2]):
                signals.append("sequential_instruction_pattern")
                score += 0.1
                break

        special_tokens = [
            "###", "---", "===", "<<<", ">>>", "```system", "```prompt",
            "[INST]", "<<SYS>>", "<|system|>", "<|user|>",
        ]
        found_tokens = [t for t in special_tokens if t in text]
        if found_tokens:
            signals.append(f"special_tokens:{','.join(found_tokens[:5])}")
            score += min(0.4, len(found_tokens) * 0.1)

        language_switches = len(re.findall(r"[\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]", text))
        if language_switches > 10:
            signals.append("multilingual_obfuscation")
            score += 0.15

        return signals, min(score, 1.0)

    def _build_explanation(
        self,
        tier_scores: dict[str, float],
        tier_labels: dict[str, str],
        signals: list[str],
    ) -> str:
        """Build human-readable explanation of the analysis."""
        if not tier_scores:
            return "No semantic injection patterns detected."

        tiers_used = list(tier_scores.keys())
        top_score = max(tier_scores.values()) if tier_scores else 0.0
        top_tier = max(tier_scores, key=tier_scores.get) if tier_scores else "unknown"

        injection_detected = any(l == "injection" for l in tier_labels.values())

        if injection_detected:
            pattern_count = len([s for s in signals if s.startswith("pattern:")])
            return (
                f"Tiered detection identified prompt injection (top tier: {top_tier}, "
                f"score: {top_score:.2f}). Tiers evaluated: {', '.join(tiers_used)}. "
                f"Found {pattern_count} pattern match(es)."
            )

        return (
            f"No semantic injection detected across {len(tiers_used)} tier(s). "
            f"Top score: {top_score:.2f} from {top_tier}."
        )