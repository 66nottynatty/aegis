"""Session management endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from aegis.api.deps import AuthDep
from aegis.core.exceptions import StorageError
from aegis.core.models import SessionCreateRequest, SessionResponse, SessionSummary
from aegis.storage.supabase import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreateRequest,
    api_key_data: AuthDep,
) -> SessionResponse:
    try:
        store = get_store()
        session = await store.create_session(
            user_id=request.user_id,
            metadata=request.metadata,
        )
        return session
    except StorageError as exc:
        logger.error("Failed to create session: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    api_key_data: AuthDep,
) -> SessionResponse:
    try:
        store = get_store()
        response = store.client.table("sessions").select("*").eq("id", session_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        row = response.data[0]
        return SessionResponse(
            session_id=row["id"],
            user_id=row.get("user_id"),
            created_at=datetime.fromisoformat(row["created_at"]),
            scan_count=row.get("scan_count", 0),
            metadata=row.get("metadata", {}),
        )
    except HTTPException:
        raise
    except StorageError as exc:
        logger.error("Storage error: %s", exc)
        raise HTTPException(status_code=500, detail="Storage error")


@router.get("/{session_id}/summary", response_model=SessionSummary)
async def get_session_summary(
    session_id: str,
    api_key_data: AuthDep,
) -> SessionSummary:
    """Get summary statistics for a session."""
    try:
        store = get_store()

        # Get session info
        session_resp = (
            store.client.table("sessions").select("*").eq("id", session_id).execute()
        )
        if not session_resp.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        session_row = session_resp.data[0]

        # Use real SQL aggregation for core stats
        stats = await store.get_session_summary(session_id)

        # Get jobs for history and complex distributions
        jobs_resp = (
            store.client.table("jobs")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )

        total_scans = stats.get("total_scans", 0)
        injections_detected = stats.get("injections_detected", 0)
        avg_risk_score = stats.get("avg_risk_score", 0.0)

        risk_distribution: dict[str, int] = {}
        agent_trigger_counts: dict[str, int] = {}
        scan_history: list[dict[str, Any]] = []
        last_scan_at: datetime | None = None

        if jobs_resp.data:
            for job in jobs_resp.data:
                result = job.get("result", {})
                if not result:
                    continue

                # Risk level distribution
                risk_level = result.get("risk_level", "unknown")
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

                # Agent triggers
                findings = result.get("findings", [])
                for finding in findings:
                    agent = finding.get("agent", "unknown")
                    if finding.get("score", 0) > 0.3:
                        agent_trigger_counts[agent] = (
                            agent_trigger_counts.get(agent, 0) + 1
                        )

                # Scan history (last 10)
                if len(scan_history) < 10:
                    created_at = job.get("created_at")
                    if created_at:
                        scan_history.append(
                            {
                                "job_id": job.get("id"),
                                "risk_level": risk_level,
                                "risk_score": result.get("risk_score", 0.0),
                                "is_injection": result.get("is_injection", False),
                                "created_at": created_at,
                            }
                        )

                # Set last scan at from first job (since we ordered by created_at desc)
                if last_scan_at is None and job.get("created_at"):
                    last_scan_at = datetime.fromisoformat(
                        job["created_at"].replace("Z", "+00:00")
                    )

        # Get top triggered agents
        top_agents = sorted(
            agent_trigger_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return SessionSummary(
            session_id=session_id,
            total_scans=total_scans,
            injections_detected=injections_detected,
            risk_distribution=risk_distribution,
            avg_risk_score=round(avg_risk_score, 4),
            top_agents_triggered=top_agents,
            scan_history=scan_history,
            created_at=datetime.fromisoformat(session_row["created_at"]),
            last_scan_at=last_scan_at,
        )

    except HTTPException:
        raise
    except StorageError as exc:
        logger.error("Storage error: %s", exc)
        raise HTTPException(status_code=500, detail="Storage error")
    except Exception as exc:
        logger.error("Failed to get session summary: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to get session summary")


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    api_key_data: AuthDep,
) -> None:
    try:
        store = get_store()
        store.client.table("sessions").delete().eq("id", session_id).execute()
    except StorageError as exc:
        logger.error("Failed to delete session: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete session")
