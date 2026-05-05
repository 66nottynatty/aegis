"""Application configuration using Pydantic Settings."""

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClassifierBackend(StrEnum):
    """Classifier backend options."""

    TIERED = "tiered"
    DEBERTA = "deberta"
    RULE = "rule"
    AEGIS = "aegis"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


class SupabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    url: str = ""
    key: str = ""
    service_key: str = ""


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    url: str = "redis://localhost:6379/0"
    password: str = ""


class BullMQSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BULLMQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    queue_name: str = "aegis-scan"


class ClassifierSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLASSIFIER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    backend: ClassifierBackend = Field(default=ClassifierBackend.TIERED, alias="backend")
    model_path: str = Field(default="/app/models/aegis_classifier.pt", alias="model_path")
    device: str = Field(default="auto", alias="device")

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v: str | ClassifierBackend) -> ClassifierBackend:
        if isinstance(v, str):
            return ClassifierBackend(v.lower())
        return v


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    timeout: int = 30


class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = ""
    public_key: str = ""
    secret_key: str = ""
    enabled: bool = False


class Mem0Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEM0_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = ""
    qdrant_host: str = ""
    qdrant_port: int = 6333
    use_supabase: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"


class RateLimitSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    free: int = 10
    pro: int = 100
    enterprise: int = 1000


class ProcessingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB")
    processing_timeout_seconds: int = Field(default=30, alias="PROCESSING_TIMEOUT_SECONDS")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_cache_ttl: int = Field(default=3600, alias="EMBEDDING_CACHE_TTL")


class WebhookSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEBHOOK_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    secret: str = ""
    max_retries: int = 3
    retry_delay_seconds: int = 60


class AppConfig:
    def __init__(self) -> None:
        self.app = Settings()
        self.supabase = SupabaseSettings()
        self.redis = RedisSettings()
        self.bullmq = BullMQSettings()
        self.classifier = ClassifierSettings()
        self.ollama = OllamaSettings()
        self.langfuse = LangfuseSettings()
        self.mem0 = Mem0Settings()
        self.rate_limit = RateLimitSettings()
        self.processing = ProcessingSettings()
        self.webhook = WebhookSettings()

    @property
    def max_file_size_bytes(self) -> int:
        return self.processing.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
