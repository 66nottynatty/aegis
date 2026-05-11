"""Prometheus metrics for Aegis Guard observability."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from fastapi import Response

# Define metrics
SCAN_COUNT = Counter(
    "aegis_scan_total", 
    "Total number of scans completed", 
    ["content_type", "risk_level"]
)

SCAN_LATENCY = Histogram(
    "aegis_scan_latency_seconds", 
    "Total time taken for scans in seconds", 
    ["content_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

INJECTION_DETECTED = Counter(
    "aegis_injection_detected_total", 
    "Total number of prompt injections detected", 
    ["content_type"]
)

AGENT_ANALYSIS_COUNT = Counter(
    "aegis_agent_analysis_total",
    "Total number of analyses performed by specific agents",
    ["agent_name", "status"] # status could be 'success', 'error', 'skipped'
)

API_REQUEST_COUNT = Counter(
    "aegis_api_request_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

def record_scan_metrics(content_type: str, risk_level: str, is_injection: bool, latency_ms: float) -> None:
    """Record metrics for a completed scan."""
    SCAN_COUNT.labels(content_type=content_type, risk_level=risk_level).inc()
    SCAN_LATENCY.labels(content_type=content_type).observe(latency_ms / 1000.0)
    if is_injection:
        INJECTION_DETECTED.labels(content_type=content_type).inc()

def get_metrics_response() -> Response:
    """Return the prometheus metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
