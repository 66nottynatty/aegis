"""DeBERTa-based prompt injection classifier.

Uses the protectai/deberta-v3-base-prompt-injection-v2 model for
improved classification accuracy on ambiguous cases.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aegis.classifiers.base import BaseClassifier, ClassifierResult

logger = logging.getLogger(__name__)

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"


class DebertaClassifier(BaseClassifier):
    """DeBERTa-based classifier for prompt injection detection."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._device = self._resolve_device(device)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_model()

    def _resolve_device(self, device: str | None) -> str:
        """Resolve compute device."""
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info("Loading DeBERTa model: %s", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info("DeBERTa model loaded successfully on %s", self._device)
        except Exception as exc:
            logger.error("Failed to load DeBERTa model: %s", exc)
            self._model = None
            self._tokenizer = None

    def is_available(self) -> bool:
        """Check if DeBERTa model is available."""
        return self._model is not None and self._tokenizer is not None

    def is_ambiguous(self, score: float) -> bool:
        """Check if a score falls in the ambiguous range.

        Ambiguous scores (0.35-0.70) indicate uncertainty that requires
        further analysis with Llama Guard.

        Args:
            score: The classification score (probability of injection)

        Returns:
            True if the score is ambiguous (between 0.35 and 0.70)
        """
        return 0.35 <= score <= 0.70

    def predict(self, text: str) -> ClassifierResult:
        """Predict prompt injection using DeBERTa.

        Args:
            text: The text content to classify

        Returns:
            ClassifierResult with score, label, confidence, and matched patterns
        """
        if not self.is_available():
            return ClassifierResult(
                score=0.0,
                label="safe",
                confidence=0.0,
                matched_patterns=["deberta_model_unavailable"],
            )

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

            injection_prob = probabilities[0, 1].item()
            safe_prob = probabilities[0, 0].item()

            label = "injection" if injection_prob > 0.5 else "safe"
            confidence = max(injection_prob, safe_prob)
            score = injection_prob

            matched_patterns = (
                [f"deberta_injection:{injection_prob:.3f}"]
                if label == "injection"
                else []
            )

            return ClassifierResult(
                score=score,
                label=label,
                confidence=confidence,
                matched_patterns=matched_patterns,
            )

        except Exception as exc:
            logger.error("DeBERTa prediction failed: %s", exc)
            return ClassifierResult(
                score=0.0,
                label="safe",
                confidence=0.0,
                matched_patterns=[f"deberta_error:{str(exc)[:50]}"],
            )

    def health_check(self) -> dict[str, Any]:
        """Return detailed health status."""
        status = super().health_check()
        status["model_name"] = self._model_name
        status["device"] = self._device
        status["max_length"] = self._max_length
        return status