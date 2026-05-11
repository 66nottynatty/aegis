"""Supabase-backed memory storage as a fallback for Mem0."""

from __future__ import annotations

import logging
from typing import Any

from aegis.storage.supabase import get_store
from aegis.storage.vector import get_vector_store

logger = logging.getLogger(__name__)


class SupabaseMemoryFallback:
    """
    A fallback memory implementation that uses Supabase pgvector for storage
    and semantic search. Matches the basic interface of Mem0.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._supabase = get_store()
        self._vector_store = get_vector_store()

    def add(
        self,
        content: str,
        user_id: str = "aegis-system",
        metadata: dict | None = None,
    ) -> None:
        """Add a memory entry to Supabase."""
        try:
            embedding = self._vector_store.embed(content)
            if not embedding:
                logger.warning("Failed to generate embedding for memory")
                return

            data = {
                "agent_name": self.agent_name,
                "user_id": user_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {},
            }

            self._supabase.client.table("aegis_agent_memories").insert(data).execute()
        except Exception as exc:
            logger.warning("Failed to store memory in Supabase: %s", exc)

    def search(
        self,
        query: str,
        user_id: str = "aegis-system",
        limit: int = 5,
    ) -> list[dict]:
        """Search for similar memories in Supabase."""
        try:
            embedding = self._vector_store.embed(query)
            if not embedding:
                return []

            response = self._supabase.client.rpc(
                "match_agent_memories",
                {
                    "query_embedding": embedding,
                    "agent_filter": self.agent_name,
                    "match_threshold": 0.5,
                    "match_count": limit,
                },
            ).execute()

            # Filter by user_id if needed (the RPC doesn't currently filter by user_id)
            results = response.data or []
            if user_id != "aegis-system":
                # If we want strict user isolation, we'd need to filter here
                # or modify the RPC.
                pass

            return results
        except Exception as exc:
            logger.warning("Failed to search memory in Supabase: %s", exc)
            return []

    def get_all(self, user_id: str = "aegis-system") -> list[dict]:
        """Retrieve all memories for a user/agent."""
        try:
            response = (
                self._supabase.client.table("aegis_agent_memories")
                .select("*")
                .eq("agent_name", self.agent_name)
                .eq("user_id", user_id)
                .execute()
            )
            return response.data or []
        except Exception as exc:
            logger.warning("Failed to fetch memories from Supabase: %s", exc)
            return []

    def delete_all(self, user_id: str = "aegis-system") -> None:
        """Delete all memories for a user/agent."""
        try:
            self._supabase.client.table("aegis_agent_memories").delete().eq(
                "agent_name", self.agent_name
            ).eq("user_id", user_id).execute()
        except Exception as exc:
            logger.warning("Failed to delete memories from Supabase: %s", exc)
