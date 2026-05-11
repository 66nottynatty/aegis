"""Tests for the Memory Agent."""

import pytest
from aegis.agents.memory_agent import MemoryAgent
from aegis.core.models import AgentFinding, ScanResult
from aegis.core.constants import ContentType, RiskLevel

@pytest.fixture
def memory_agent():
    return MemoryAgent(enable_memory=False)

def test_memory_agent_initialization(memory_agent):
    assert memory_agent.name.value == "memory"

def test_memory_agent_analyze(memory_agent):
    scan_result = ScanResult(
        job_id="test-job",
        risk_level=RiskLevel.HIGH,
        risk_score=0.8,
        is_injection=True,
        confidence=0.9,
        findings=[],
        summary="Injection detected",
        content_type=ContentType.TEXT,
        processing_time_ms=100
    )
    
    context = {
        "scan_result": scan_result,
        "session_id": "test-session"
    }
    
    finding = memory_agent.analyze("", context)
    
    assert isinstance(finding, AgentFinding)
    assert finding.agent == "memory"
    assert "memories_stored" in finding.explanation
