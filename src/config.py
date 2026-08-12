from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: SecretStr

    llm_api_key: SecretStr
    llm_base_url: str = "https://api.gonkagate.com/v1"
    llm_model: str = "gpt-5.6"

    app_timezone: str = "Europe/Moscow"
