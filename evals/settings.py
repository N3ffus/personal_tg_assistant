from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
JUDGE_MODEL = "zai-org/GLM-5.3-Flash"


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.eval"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepinfra_api_key: SecretStr
    eval_judge_model: str = JUDGE_MODEL
    llm_api_key: SecretStr
    llm_base_url: str = "https://api.gonkagate.com/v1"
    llm_model: str = "gpt-5.6"
    eval_timeout_seconds: float = Field(default=90, gt=0, le=300)

    @field_validator("deepinfra_api_key", "llm_api_key")
    @classmethod
    def reject_empty_keys(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API key must not be empty")
        return value

    @field_validator("eval_judge_model", "llm_model", "llm_base_url")
    @classmethod
    def reject_empty_settings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def load_eval_settings() -> EvalSettings:
    # BaseSettings obtains required fields from the environment at runtime.
    return EvalSettings()  # type: ignore[call-arg]
