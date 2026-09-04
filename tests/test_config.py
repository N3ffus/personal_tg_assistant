import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from src.config import Settings


def valid_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "telegram_bot_token": "telegram-token",
        "telegram_allowed_user_id": 42,
        "llm_api_key": "llm-key",
        "linear_api_key": "linear-key",
        "linear_team_id": "linear-team",
        "google_oauth_client_id": "google-client",
        "google_oauth_client_secret": "google-secret",
        "google_oauth_redirect_uri": "https://example.test/oauth/google/callback",
        "google_token_encryption_key": Fernet.generate_key().decode(),
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_settings_apply_safe_application_defaults() -> None:
    settings = valid_settings()

    assert settings.llm_base_url == "https://api.gonkagate.com/v1"
    assert settings.llm_model == "gpt-5.6"
    assert settings.app_timezone == "Europe/Moscow"
    assert settings.database_path == "data/assistant.db"
    assert settings.http_host == "0.0.0.0"
    assert settings.http_port == 8080
    assert settings.telegram_bot_token.get_secret_value() == "telegram-token"
    assert settings.telegram_allowed_user_id == 42
    assert settings.linear_api_key.get_secret_value() == "linear-key"
    assert settings.linear_team_id == "linear-team"
    assert settings.google_oauth_client_secret.get_secret_value() == "google-secret"


def test_settings_accept_runtime_overrides() -> None:
    settings = valid_settings(
        app_timezone="Asia/Tokyo",
        database_path="state/test.db",
        http_host="127.0.0.1",
        http_port=9000,
    )

    assert settings.app_timezone == "Asia/Tokyo"
    assert settings.database_path == "state/test.db"
    assert settings.http_host == "127.0.0.1"
    assert settings.http_port == 9000


def test_settings_require_integration_credentials() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)  # type: ignore[call-arg]

    missing_fields = {error["loc"][0] for error in exc.value.errors()}
    assert missing_fields == {
        "telegram_bot_token",
        "telegram_allowed_user_id",
        "llm_api_key",
        "linear_api_key",
        "linear_team_id",
        "google_oauth_client_id",
        "google_oauth_client_secret",
        "google_oauth_redirect_uri",
        "google_token_encryption_key",
    }


@pytest.mark.parametrize(
    "field",
    [
        "telegram_bot_token",
        "llm_api_key",
        "llm_base_url",
        "llm_model",
        "linear_api_key",
        "linear_team_id",
        "google_oauth_client_id",
        "google_oauth_client_secret",
        "google_oauth_redirect_uri",
        "database_path",
        "http_host",
    ],
)
def test_settings_reject_blank_required_values(field: str) -> None:
    with pytest.raises(ValidationError):
        valid_settings(**{field: "   "})


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"telegram_allowed_user_id": 0}, "telegram_allowed_user_id"),
        ({"http_port": 0}, "http_port"),
        ({"http_port": 65536}, "http_port"),
        ({"app_timezone": "Mars/Olympus"}, "app_timezone"),
        ({"llm_base_url": "not-a-url"}, "llm_base_url"),
        (
            {"google_oauth_redirect_uri": "http://assistant.example/callback"},
            "google_oauth_redirect_uri",
        ),
        (
            {"google_token_encryption_key": "not-a-fernet-key"},
            "google_token_encryption_key",
        ),
    ],
)
def test_settings_reject_unsafe_runtime_values(
    overrides: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValidationError) as exc:
        valid_settings(**overrides)

    assert exc.value.errors()[0]["loc"] == (field,)


def test_settings_allow_plain_http_only_for_loopback_services() -> None:
    settings = valid_settings(
        llm_base_url="http://127.0.0.1:8000/v1/",
        google_oauth_redirect_uri="http://localhost:8080/oauth/google/callback/",
    )

    assert settings.llm_base_url == "http://127.0.0.1:8000/v1/"
    assert (
        settings.google_oauth_redirect_uri
        == "http://localhost:8080/oauth/google/callback/"
    )
