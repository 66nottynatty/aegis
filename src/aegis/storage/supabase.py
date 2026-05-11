"""Supabase client wrapper for job storage and session management."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from aegis.core.config import get_config
from aegis.core.constants import JobStatus
from aegis.core.exceptions import JobNotFoundError, StorageError
from aegis.core.models import FeedbackRequest, JobResponse, ScanResult, SessionResponse

logger = logging.getLogger(__name__)


class SupabaseStore:
    def __init__(self) -> None:
        self._config = get_config()
        self._client: Any | None = None

    def connect(self) -> None:
        try:
            from supabase import create_client

            self._client = create_client(
                self._config.supabase.url,
                self._config.supabase.service_key or self._config.supabase.key,
            )
            logger.info("Supabase client connected")
        except Exception as exc:
            logger.warning("Supabase connection failed: %s", exc)
            self._client = None

    @property
    def client(self) -> Any:
        if not self._client:
            raise StorageError("Supabase not connected. Call connect() first.")
        return self._client

    async def create_job(self, scan_request_data: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        try:
            self.client.table("jobs").insert({
                "id": job_id,
                "status": JobStatus.PENDING.value,
                "request": scan_request_data,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as exc:
            raise StorageError(f"Failed to create job: {exc}") from exc
        return job_id

    async def update_job_status(self, job_id: str, status: JobStatus, result: dict | None = None, error: str | None = None) -> None:
        update_data: dict[str, Any] = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if result is not None:
            update_data["result"] = result
        if error is not None:
            update_data["error"] = error

        try:
            response = self.client.table("jobs").update(update_data).eq("id", job_id).execute()
            if not response.data:
                raise JobNotFoundError(job_id)
        except JobNotFoundError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to update job {job_id}: {exc}") from exc

    async def get_job(self, job_id: str) -> JobResponse:
        try:
            response = self.client.table("jobs").select("*").eq("id", job_id).execute()
            if not response.data:
                raise JobNotFoundError(job_id)
            row = response.data[0]
            return self._row_to_job_response(row)
        except JobNotFoundError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to get job {job_id}: {exc}") from exc

    async def create_session(self, user_id: str | None = None, metadata: dict | None = None) -> SessionResponse:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        try:
            self.client.table("sessions").insert({
                "id": session_id,
                "user_id": user_id,
                "metadata": metadata or {},
                "scan_count": 0,
                "created_at": now.isoformat(),
            }).execute()
        except Exception as exc:
            raise StorageError(f"Failed to create session: {exc}") from exc

        return SessionResponse(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            scan_count=0,
            metadata=metadata or {},
        )

    async def increment_session_scan_count(self, session_id: str) -> None:
        try:
            self.client.rpc("increment_session_scan_count", {"sid": session_id}).execute()
        except Exception as exc:
            logger.warning("Failed to increment session scan count: %s", exc)

    async def store_feedback(self, feedback: FeedbackRequest, api_key_id: str) -> str:
        feedback_id = str(uuid.uuid4())
        try:
            self.client.table("feedback").insert({
                "id": feedback_id,
                "job_id": feedback.job_id,
                "is_correct": feedback.is_correct,
                "actual_risk_level": feedback.actual_risk_level.value if feedback.actual_risk_level else None,
                "notes": feedback.notes,
                "api_key_id": api_key_id,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as exc:
            raise StorageError(f"Failed to store feedback: {exc}") from exc
        return feedback_id

    async def get_api_key(self, key_hash: str) -> dict[str, Any] | None:
        try:
            response = self.client.table("api_keys").select("*").eq("key_hash", key_hash).eq("is_active", True).execute()
            return response.data[0] if response.data else None
        except Exception as exc:
            logger.warning("Failed to get API key: %s", exc)
            return None

    async def create_api_key(self, user_id: str, tier: str = "free", description: str | None = None, is_admin: bool = False) -> dict[str, Any]:
        import hashlib
        import secrets
        
        raw_key = f"aegis_{tier}_{secrets.token_hex(16)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        data = {
            "key_hash": key_hash,
            "user_id": user_id,
            "tier": tier,
            "description": description,
            "is_admin": is_admin,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        try:
            response = self.client.table("api_keys").insert(data).execute()
            result = response.data[0]
            result["raw_key"] = raw_key  # Return the raw key only once upon creation
            return result
        except Exception as exc:
            raise StorageError(f"Failed to create API key: {exc}") from exc

    async def list_api_keys(self) -> list[dict[str, Any]]:
        try:
            response = self.client.table("api_keys").select("*").order("created_at", desc=True).execute()
            return response.data
        except Exception as exc:
            raise StorageError(f"Failed to list API keys: {exc}") from exc

    async def revoke_api_key(self, key_id: str) -> bool:
        try:
            response = self.client.table("api_keys").update({"is_active": False, "updated_at": datetime.utcnow().isoformat()}).eq("id", key_id).execute()
            return len(response.data) > 0
        except Exception as exc:
            # Check if updated_at exists, some schemas might not have it for api_keys
            try:
                response = self.client.table("api_keys").update({"is_active": False}).eq("id", key_id).execute()
                return len(response.data) > 0
            except Exception:
                raise StorageError(f"Failed to revoke API key: {exc}") from exc

    def _row_to_job_response(self, row: dict[str, Any]) -> JobResponse:
        result = None
        if row.get("result"):
            result = ScanResult(**row["result"])

        return JobResponse(
            job_id=row["id"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else None,
            result=result,
            error=row.get("error"),
        )


_store_instance: SupabaseStore | None = None


def get_store() -> SupabaseStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = SupabaseStore()
        _store_instance.connect()
    return _store_instance
