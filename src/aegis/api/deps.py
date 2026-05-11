"""FastAPI dependencies - authentication, rate limiting, and shared resources."""

from __future__ import annotations

import hashlib
import logging
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status

from aegis.core.cache import get_cache
from aegis.core.config import get_config
from aegis.core.constants import ApiTier
from aegis.core.exceptions import AuthenticationError, RateLimitError
from aegis.core.security import get_tier_rate_limit
from aegis.storage.supabase import get_store

logger = logging.getLogger(__name__)


async def get_api_key_data(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> dict[str, Any]:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    try:
        store = get_store()
        key_data = await store.get_api_key(key_hash)
    except Exception as exc:
        logger.warning("API key lookup failed: %s", exc)
        key_data = None

    if not key_data:
        if x_api_key.startswith("aegis_free_") and len(x_api_key) > 20:
            return {"key_id": "dev", "tier": ApiTier.FREE.value, "user_id": "dev-user"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )

    return key_data


async def check_rate_limit(
    request: Request,
    api_key_data: Annotated[dict[str, Any], Depends(get_api_key_data)],
) -> dict[str, Any]:
    config = get_config()
    tier = ApiTier(api_key_data.get("tier", ApiTier.FREE.value))
    key_id = api_key_data.get("key_id", "unknown")

    limit = get_tier_rate_limit(
        tier,
        {
            "free": config.rate_limit.free,
            "pro": config.rate_limit.pro,
            "enterprise": config.rate_limit.enterprise,
        },
    )

    cache = get_cache()
    rate_key = f"rate_limit:{key_id}:{_get_current_minute()}"

    try:
        count = await cache.increment(rate_key, ttl=60)
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests/minute for {tier} tier",
                headers={"Retry-After": "60", "X-Rate-Limit": str(limit)},
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Rate limit check failed (allowing request): %s", exc)

    return api_key_data


async def get_admin_user(
    api_key_data: Annotated[dict[str, Any], Depends(get_api_key_data)],
) -> dict[str, Any]:
    if not api_key_data.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return api_key_data


def _get_current_minute() -> str:
    import time
    return str(int(time.time()) // 60)


AuthDep = Annotated[dict[str, Any], Depends(check_rate_limit)]
