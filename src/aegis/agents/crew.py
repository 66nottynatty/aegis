"""Crew orchestrator - coordinates all agents for content scanning using LangGraph."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from aegis.core.constants import RiskLevel
from aegis.core.models import AgentFinding, ScanRequest, ScanResult
from aegis.pipeline.graph import get_graph
from aegis.pipeline.state import AegisState

logger = logging.getLogger(__name__)


class AegisCrew:
    """Main orchestrator for Aegis content analysis."""

    def __init__(self, enable_memory: bool = True) -> None:
        self._enable_memory = enable_memory
        self._graph = get_graph()

    def scan(self, request: ScanRequest, job_id: str | None = None) -> ScanResult:
        """Execute a scan using the LangGraph pipeline."""
        job_id = job_id or str(uuid.uuid4())
        start_time = time.monotonic()

        try:
            # Build initial state
            state: AegisState = {
                "content": request.content,
                "content_type": request.content_type,
                "session_id": request.session_id,
                "metadata": request.metadata or {},
                "extracted_text": "",
                "processed_data": {},
                "guardrails_result": {},
                "sanitized_content": request.content,
                "should_block": False,
                "structural_finding": None,
                "semantic_finding": None,
                "intent_finding": None,
                "visual_finding": None,
                "behavioral_finding": None,
                "findings": [],
                "verdict_finding": None,
                "risk_score": 0.0,
                "risk_level": RiskLevel.SAFE.value,
                "is_injection": False,
                "confidence": 0.0,
                "summary": "",
                "job_id": job_id,
                "processing_time_ms": 0,
                "error": None,
            }

            # Execute the graph
            final_state = self._graph.invoke(state)

            # Update processing time
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            final_state["processing_time_ms"] = elapsed_ms

            # Record metrics
            try:
                from aegis.core.metrics import record_scan_metrics
                record_scan_metrics(
                    content_type=final_state["content_type"].value,
                    risk_level=final_state["risk_level"],
                    is_injection=final_state["is_injection"],
                    latency_ms=float(elapsed_ms),
                )
            except Exception as m_exc:
                logger.debug("Failed to record metrics: %s", m_exc)

            # Build and return result
            return self._build_scan_result(final_state)

        except Exception as exc:
            logger.error("Crew scan failed for job %s: %s", job_id, exc, exc_info=True)
            elapsed = int((time.monotonic() - start_time) * 1000)
            return ScanResult(
                job_id=job_id,
                risk_level=RiskLevel.SAFE,
                risk_score=0.0,
                is_injection=False,
                confidence=0.0,
                findings=[],
                summary=f"Analysis failed: {exc}",
                content_type=request.content_type,
                processing_time_ms=elapsed,
            )

    async def scan_async(self, request: ScanRequest, job_id: str | None = None) -> ScanResult:
        """Execute scan asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, self.scan, request, job_id)

    def _build_scan_result(self, state: AegisState) -> ScanResult:
        """Build ScanResult from final state."""
        from aegis.core.constants import RiskLevel

        # Collect all findings
        findings: list[AgentFinding] = []
        for key in ["structural_finding", "semantic_finding", "intent_finding", "visual_finding", "behavioral_finding"]:
            finding = state.get(key)
            if finding and isinstance(finding, AgentFinding):
                findings.append(finding)

        # Add verdict finding
        verdict = state.get("verdict_finding")
        if verdict and isinstance(verdict, AgentFinding):
            findings.append(verdict)

        risk_level_str = state.get("risk_level", "safe")
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.SAFE

        return ScanResult(
            job_id=state.get("job_id", "unknown"),
            risk_level=risk_level,
            risk_score=state.get("risk_score", 0.0),
            is_injection=state.get("is_injection", False),
            confidence=state.get("confidence", 0.0),
            findings=findings,
            summary=state.get("summary", ""),
            content_type=state["content_type"],
            processing_time_ms=state.get("processing_time_ms", 0),
            sanitized_content=state.get("sanitized_content"),
        )
