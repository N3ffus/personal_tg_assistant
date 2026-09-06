import hashlib
import json
from base64 import urlsafe_b64encode
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import requests
from cryptography.fernet import Fernet
from google_auth_oauthlib.flow import Flow

from src.infrastructure.calendar.google import SCOPES
from src.infrastructure.calendar.oauth import GoogleOAuthService, create_oauth_app
from src.infrastructure.calendar.storage import CalendarStorage


def oauth_service(storage: object) -> GoogleOAuthService:
    return GoogleOAuthService(
        storage=storage,  # type: ignore[arg-type]
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://assistant.example/oauth/google/callback",
    )


@pytest.mark.asyncio
async def test_pkce_verifier_survives_restart_and_matches_authorization_challenge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = str(tmp_path / "calendar.db")
    encryption_key = Fernet.generate_key().decode()
    storage = CalendarStorage(
        database_path=database_path, encryption_key=encryption_key
    )
    await storage.initialize()
    url = await oauth_service(storage).authorization_url(user_id=42)
    query = parse_qs(urlsplit(url).query)
    assert query["code_challenge_method"] == ["S256"]

    def exchange(
        session: requests.Session,
        request: requests.PreparedRequest,
        **kwargs: object,
    ) -> requests.Response:
        assert request.url == "https://oauth2.googleapis.com/token"
        assert isinstance(request.body, str)
        body = parse_qs(request.body)
        verifier = body.get("code_verifier", [""])[0]
        assert 43 <= len(verifier) <= 128, "Token request is missing the PKCE verifier"
        challenge = urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        assert challenge.decode().rstrip("=") == query["code_challenge"][0]
        assert verifier not in url
        assert body["code"] == ["authorization-code"]
        response = requests.Response()
        response.request = request
        response.status_code = 200
        response._content = json.dumps(
            {
                "access_token": "test-access-token",
                "refresh_token": "test-refresh-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": " ".join(SCOPES),
            }
        ).encode()
        return response

    send = Mock(side_effect=exchange)
    monkeypatch.setattr(
        requests.Session, "send", lambda self, request, **kw: send(self, request, **kw)
    )
    restarted_storage = CalendarStorage(
        database_path=database_path, encryption_key=encryption_key
    )
    await restarted_storage.initialize()
    restarted_service = oauth_service(restarted_storage)
    assert await restarted_service.complete(
        code="authorization-code", state=query["state"][0]
    )
    credentials = await restarted_storage.load_credentials(user_id=42)
    assert credentials is not None
    assert credentials["refresh_token"] == "test-refresh-token"
    assert not await restarted_service.complete(
        code="authorization-code", state=query["state"][0]
    )
    assert send.call_count == 1


@pytest.mark.asyncio
async def test_authorization_url_binds_google_state_to_telegram_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = SimpleNamespace(create_oauth_state=AsyncMock(return_value="state-123"))
    flow = SimpleNamespace(
        authorization_url=Mock(
            return_value=("https://accounts.google.test/authorize", "state-123")
        )
    )
    from_client_config = Mock(return_value=flow)
    monkeypatch.setattr(Flow, "from_client_config", from_client_config)

    result = await oauth_service(storage).authorization_url(user_id=42)

    assert result == "https://accounts.google.test/authorize"
    verifier = storage.create_oauth_state.call_args.kwargs["code_verifier"]
    assert 43 <= len(verifier) <= 128
    storage.create_oauth_state.assert_awaited_once_with(
        user_id=42, code_verifier=verifier
    )
    flow.authorization_url.assert_called_once_with(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state="state-123",
    )
    config = from_client_config.call_args.args[0]["web"]
    assert config == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["https://assistant.example/oauth/google/callback"],
    }
    assert from_client_config.call_args.kwargs == {
        "scopes": SCOPES,
        "redirect_uri": "https://assistant.example/oauth/google/callback",
        "state": None,
        "code_verifier": verifier,
        "autogenerate_code_verifier": False,
    }


@pytest.mark.asyncio
async def test_complete_rejects_unknown_or_expired_state_without_token_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = SimpleNamespace(
        consume_oauth_state=AsyncMock(return_value=None),
        save_credentials=AsyncMock(),
    )
    from_client_config = Mock()
    monkeypatch.setattr(Flow, "from_client_config", from_client_config)

    completed = await oauth_service(storage).complete(code="code", state="bad-state")

    assert completed is False
    storage.consume_oauth_state.assert_awaited_once_with(state="bad-state")
    storage.save_credentials.assert_not_awaited()
    from_client_config.assert_not_called()


@pytest.mark.asyncio
async def test_complete_exchanges_code_and_persists_full_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = SimpleNamespace(
        consume_oauth_state=AsyncMock(return_value=(42, "saved-verifier")),
        save_credentials=AsyncMock(),
    )
    payload = {
        "token": "access-token",
        "refresh_token": "refresh-token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": SCOPES,
        "expiry": "2026-09-03T12:00:00Z",
    }
    flow = SimpleNamespace(
        fetch_token=Mock(),
        credentials=SimpleNamespace(to_json=Mock(return_value=json.dumps(payload))),
    )
    from_client_config = Mock(return_value=flow)
    monkeypatch.setattr(Flow, "from_client_config", from_client_config)

    completed = await oauth_service(storage).complete(
        code="authorization-code",
        state="state-123",
    )

    assert completed is True
    flow.fetch_token.assert_called_once_with(code="authorization-code")
    assert from_client_config.call_args.kwargs["state"] == "state-123"
    assert from_client_config.call_args.kwargs["code_verifier"] == "saved-verifier"
    storage.save_credentials.assert_awaited_once_with(
        user_id=42,
        credentials=payload,
    )


@pytest.mark.asyncio
async def test_complete_rejects_non_object_credentials_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = SimpleNamespace(
        consume_oauth_state=AsyncMock(return_value=(42, "saved-verifier")),
        save_credentials=AsyncMock(),
    )
    flow = SimpleNamespace(
        fetch_token=Mock(),
        credentials=SimpleNamespace(to_json=Mock(return_value='["token"]')),
    )
    monkeypatch.setattr(
        Flow,
        "from_client_config",
        Mock(return_value=flow),
    )

    with pytest.raises(TypeError, match="serialize to an object"):
        await oauth_service(storage).complete(code="code", state="state-123")

    storage.save_credentials.assert_not_awaited()


async def request(app: object, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_health_endpoint_reports_ready() -> None:
    oauth = SimpleNamespace(complete=AsyncMock())

    response = await request(create_oauth_app(oauth=oauth), "/health")  # type: ignore[arg-type]

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "",
        "?code=code",
        "?state=state",
        "?code=code&state=state&error=access_denied",
    ],
)
async def test_callback_handles_cancelled_or_incomplete_request(query: str) -> None:
    oauth = SimpleNamespace(complete=AsyncMock())

    response = await request(
        create_oauth_app(oauth=oauth),  # type: ignore[arg-type]
        f"/oauth/google/callback{query}",
    )

    assert response.status_code == 200
    assert "Подключение Google Calendar отменено" in response.text
    oauth.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_rejects_consumed_or_expired_state() -> None:
    oauth = SimpleNamespace(complete=AsyncMock(return_value=False))

    response = await request(
        create_oauth_app(oauth=oauth),  # type: ignore[arg-type]
        "/oauth/google/callback?code=code&state=expired",
    )

    assert response.status_code == 400
    assert "недействительна или устарела" in response.text
    oauth.complete.assert_awaited_once_with(code="code", state="expired")


@pytest.mark.asyncio
async def test_callback_reports_successful_connection() -> None:
    oauth = SimpleNamespace(complete=AsyncMock(return_value=True))

    response = await request(
        create_oauth_app(oauth=oauth),  # type: ignore[arg-type]
        "/oauth/google/callback?code=code&state=state-123",
    )

    assert response.status_code == 200
    assert "Google Calendar подключён" in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    oauth.complete.assert_awaited_once_with(code="code", state="state-123")


@pytest.mark.asyncio
async def test_callback_hides_token_exchange_failure() -> None:
    oauth = SimpleNamespace(complete=AsyncMock(side_effect=RuntimeError("secret")))

    response = await request(
        create_oauth_app(oauth=oauth),  # type: ignore[arg-type]
        "/oauth/google/callback?code=code&state=state-123",
    )

    assert response.status_code == 502
    assert "Попробуйте снова" in response.text
    assert "secret" not in response.text
