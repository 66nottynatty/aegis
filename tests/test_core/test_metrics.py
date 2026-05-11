"""Tests for Prometheus metrics module."""

import pytest

from aegis.core.metrics import (
    ACTIVE_SESSIONS,
    AGENT_ANALYSIS_COUNT,
    API_REQUEST_COUNT,
    CLASSIFIER_REQUESTS,
    INJECTION_DETECTED,
    QUEUE_DEPTH,
    SCAN_COUNT,
    SCAN_LATENCY,
    record_agent_metrics,
    record_api_metrics,
    record_classifier_metrics,
    record_scan_metrics,
)


class TestScanMetrics:
    """Test scan metrics recording."""

    def test_record_scan_metrics_increments_count(self) -> None:
        """Test that scan metrics increment the counter."""
        initial_value = SCAN_COUNT.labels(content_type="text", risk_level="safe")._value.get()
        record_scan_metrics("text", "safe", False, 100.0)
        new_value = SCAN_COUNT.labels(content_type="text", risk_level="safe")._value.get()
        assert new_value == initial_value + 1

    def test_record_scan_metrics_injection_detected(self) -> None:
        """Test that injection detection increments counter."""
        initial_value = INJECTION_DETECTED.labels(content_type="text")._value.get()
        record_scan_metrics("text", "high", True, 100.0)
        new_value = INJECTION_DETECTED.labels(content_type="text")._value.get()
        assert new_value == initial_value + 1

    def test_record_scan_metrics_records_latency(self) -> None:
        """Test that scan latency is recorded."""
        # Latency is recorded as an observation, we can't easily test the histogram directly
        # but we can verify the function doesn't raise
        record_scan_metrics("text", "safe", False, 500.0)  # 500ms
        record_scan_metrics("html", "high", True, 2500.0)  # 2.5s


class TestAgentMetrics:
    """Test agent metrics recording."""

    def test_record_agent_metrics_success(self) -> None:
        """Test recording successful agent analysis."""
        initial_value = AGENT_ANALYSIS_COUNT.labels(agent_name="semantic", status="success")._value.get()
        record_agent_metrics("semantic", "success")
        new_value = AGENT_ANALYSIS_COUNT.labels(agent_name="semantic", status="success")._value.get()
        assert new_value == initial_value + 1

    def test_record_agent_metrics_error(self) -> None:
        """Test recording failed agent analysis."""
        initial_value = AGENT_ANALYSIS_COUNT.labels(agent_name="intent", status="error")._value.get()
        record_agent_metrics("intent", "error")
        new_value = AGENT_ANALYSIS_COUNT.labels(agent_name="intent", status="error")._value.get()
        assert new_value == initial_value + 1

    def test_record_agent_metrics_skipped(self) -> None:
        """Test recording skipped agent analysis."""
        initial_value = AGENT_ANALYSIS_COUNT.labels(agent_name="visual", status="skipped")._value.get()
        record_agent_metrics("visual", "skipped")
        new_value = AGENT_ANALYSIS_COUNT.labels(agent_name="visual", status="skipped")._value.get()
        assert new_value == initial_value + 1


class TestAPIMetrics:
    """Test API metrics recording."""

    def test_record_api_metrics(self) -> None:
        """Test recording API request metrics."""
        initial_value = API_REQUEST_COUNT.labels(method="POST", endpoint="/v1/scan", status_code="200")._value.get()
        record_api_metrics("POST", "/v1/scan", 200)
        new_value = API_REQUEST_COUNT.labels(method="POST", endpoint="/v1/scan", status_code="200")._value.get()
        assert new_value == initial_value + 1

    def test_record_api_metrics_error(self) -> None:
        """Test recording API error metrics."""
        initial_value = API_REQUEST_COUNT.labels(method="POST", endpoint="/v1/scan", status_code="500")._value.get()
        record_api_metrics("POST", "/v1/scan", 500)
        new_value = API_REQUEST_COUNT.labels(method="POST", endpoint="/v1/scan", status_code="500")._value.get()
        assert new_value == initial_value + 1


class TestClassifierMetrics:
    """Test classifier metrics recording."""

    def test_record_classifier_metrics_safe(self) -> None:
        """Test recording safe classifier result."""
        initial_value = CLASSIFIER_REQUESTS.labels(classifier="rule_based", result="safe")._value.get()
        record_classifier_metrics("rule_based", "safe")
        new_value = CLASSIFIER_REQUESTS.labels(classifier="rule_based", result="safe")._value.get()
        assert new_value == initial_value + 1

    def test_record_classifier_metrics_injection(self) -> None:
        """Test recording injection classifier result."""
        initial_value = CLASSIFIER_REQUESTS.labels(classifier="deberta", result="injection")._value.get()
        record_classifier_metrics("deberta", "injection")
        new_value = CLASSIFIER_REQUESTS.labels(classifier="deberta", result="injection")._value.get()
        assert new_value == initial_value + 1

    def test_record_classifier_metrics_error(self) -> None:
        """Test recording error classifier result."""
        initial_value = CLASSIFIER_REQUESTS.labels(classifier="llama_guard", result="error")._value.get()
        record_classifier_metrics("llama_guard", "error")
        new_value = CLASSIFIER_REQUESTS.labels(classifier="llama_guard", result="error")._value.get()
        assert new_value == initial_value + 1


class TestGaugeMetrics:
    """Test gauge metrics."""

    def test_active_sessions_gauge(self) -> None:
        """Test active sessions gauge."""
        ACTIVE_SESSIONS.set(5)
        assert ACTIVE_SESSIONS._value.get() == 5

    def test_queue_depth_gauge(self) -> None:
        """Test queue depth gauge."""
        QUEUE_DEPTH.set(10)
        assert QUEUE_DEPTH._value.get() == 10