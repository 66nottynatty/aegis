"""Admin routes for API key management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from aegis.api.deps import get_admin_user
from aegis.storage.supabase import get_store

router = APIRouter(prefix="/admin", tags=["Admin"])


class ApiKeyCreateRequest(BaseModel):
    user_id: str
    tier: str = Field(default="free", pattern="^(free|pro|enterprise)$")
    description: str | None = None
    is_admin: bool = False


@router.post("/keys", status_code=status.HTTP_201_CREATED)
async def create_key(
    request: ApiKeyCreateRequest,
    admin_user: dict[str, Any] = Depends(get_admin_user),
) -> dict[str, Any]:
    """Create a new API key (Admin only)."""
    store = get_store()
    try:
        return await store.create_api_key(
            user_id=request.user_id,
            tier=request.tier,
            description=request.description,
            is_admin=request.is_admin,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {exc}",
        )


@router.get("/keys")
async def list_keys(
    admin_user: dict[str, Any] = Depends(get_admin_user),
) -> list[dict[str, Any]]:
    """List all API keys (Admin only)."""
    store = get_store()
    try:
        return await store.list_api_keys()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {exc}",
        )


@router.delete("/keys/{key_id}")
async def revoke_key(
    key_id: str,
    admin_user: dict[str, Any] = Depends(get_admin_user),
) -> dict[str, str]:
    """Revoke an API key (Admin only)."""
    store = get_store()
    try:
        success = await store.revoke_api_key(key_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        return {"status": "success", "message": "API key revoked"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke API key: {exc}",
        )
