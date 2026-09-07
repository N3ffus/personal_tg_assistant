from typing import Literal, Self
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cryptography.fernet import Fernet
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: SecretStr
    telegram_allowed_user_id: int = Field(gt=0)

    llm_api_key: SecretStr
    llm_base_url: str = "https://api.gonkagate.com/v1"
    llm_model: str = "gpt-5.6"

    app_timezone: str = "Europe/Moscow"

    linear_api_key: SecretStr
    linear_team_id: str

    google_oauth_client_id: str
    google_oauth_client_secret: SecretStr
    google_oauth_redirect_uri: str
    google_token_encryption_key: SecretStr
    database_path: str = "data/assistant.db"
    # The OAuth callback is intentionally reachable through the container port.
    http_host: str = "0.0.0.0"  # nosec B104
    http_port: int = Field(default=8080, ge=1, le=65535)

    # Knowledge memory is optional: the bot keeps working when Neo4j is down.
    knowledge_memory_enabled: bool = False
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    # Graphiti falls back to the main LLM provider unless a cheaper one is set.
    graphiti_llm_model: str = ""
    graphiti_llm_api_key: SecretStr = SecretStr("")
    graphiti_llm_base_url: str = ""
    graphiti_llm_structured_output: Literal["json_schema", "json_object"] = (
        "json_schema"
    )
    graphiti_embedding_model: str = "text-embedding-3-small"
    graphiti_embedding_api_key: SecretStr = SecretStr("")
    graphiti_embedding_base_url: str = ""
    graphiti_embedding_dim: int = Field(default=1536, ge=1, le=8192)
    # Graphiti fans out to 20 concurrent LLM calls by default, which is far
    # more memory than a small host can spare while the bot keeps polling.
    graphiti_max_coroutines: int = Field(default=4, ge=1, le=50)

    @property
    def resolved_graphiti_llm_model(self) -> str:
        return self.graphiti_llm_model or self.llm_model

    @property
    def resolved_graphiti_llm_api_key(self) -> str:
        key = self.graphiti_llm_api_key.get_secret_value()
        return key or self.llm_api_key.get_secret_value()

    @property
    def resolved_graphiti_llm_base_url(self) -> str:
        return self.graphiti_llm_base_url or self.llm_base_url

    @property
    def resolved_graphiti_embedding_api_key(self) -> str:
        key = self.graphiti_embedding_api_key.get_secret_value()
        return key or self.resolved_graphiti_llm_api_key

    @property
    def resolved_graphiti_embedding_base_url(self) -> str:
        return self.graphiti_embedding_base_url or self.resolved_graphiti_llm_base_url

    @field_validator(
        "llm_base_url",
        "llm_model",
        "app_timezone",
        "linear_team_id",
        "google_oauth_client_id",
        "google_oauth_redirect_uri",
        "database_path",
        "http_host",
        "neo4j_uri",
        "neo4j_user",
        "graphiti_embedding_model",
    )
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator(
        "telegram_bot_token",
        "llm_api_key",
        "linear_api_key",
        "google_oauth_client_secret",
        "google_token_encryption_key",
    )
    @classmethod
    def reject_blank_secrets(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("neo4j_uri")
    @classmethod
    def require_bolt_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {
            "bolt",
            "bolt+s",
            "bolt+ssc",
            "neo4j",
            "neo4j+s",
            "neo4j+ssc",
        }:
            raise ValueError("must be a bolt:// or neo4j:// URL")
        if not parsed.hostname:
            raise ValueError("must include a host")
        return value

    @model_validator(mode="after")
    def require_knowledge_credentials(self) -> Self:
        if (
            self.knowledge_memory_enabled
            and not self.neo4j_password.get_secret_value().strip()
        ):
            raise ValueError(
                "neo4j_password is required when knowledge memory is enabled"
            )
        return self

    @field_validator("llm_base_url", "google_oauth_redirect_uri")
    @classmethod
    def require_secure_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
        }:
            raise ValueError("must use HTTPS outside localhost")
        return value

    @field_validator("app_timezone")
    @classmethod
    def require_known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("must be a valid IANA timezone") from error
        return value

    @field_validator("google_token_encryption_key")
    @classmethod
    def require_fernet_key(cls, value: SecretStr) -> SecretStr:
        try:
            Fernet(value.get_secret_value().encode())
        except (TypeError, ValueError) as error:
            raise ValueError("must be a valid Fernet key") from error
        return value
