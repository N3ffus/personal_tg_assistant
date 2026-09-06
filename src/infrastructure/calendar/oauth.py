import asyncio
import json
import logging
import secrets

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from src.infrastructure.calendar.google import SCOPES
from src.infrastructure.calendar.storage import CalendarStorage

logger = logging.getLogger(__name__)
GOOGLE_OAUTH_EXCHANGE_URL = "https://oauth2.googleapis.com/token"
OAUTH_RESPONSE_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


class GoogleOAuthService:
    def __init__(
        self,
        *,
        storage: CalendarStorage,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        self._storage = storage
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    async def authorization_url(self, *, user_id: int) -> str:
        code_verifier = secrets.token_urlsafe(64)
        state = await self._storage.create_oauth_state(
            user_id=user_id, code_verifier=code_verifier
        )
        flow = self._flow(code_verifier=code_verifier)
        url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )
        return str(url)

    async def complete(self, *, code: str, state: str) -> bool:
        oauth_state = await self._storage.consume_oauth_state(state=state)
        if oauth_state is None:
            return False
        user_id, code_verifier = oauth_state
        flow = self._flow(state=state, code_verifier=code_verifier)
        await asyncio.to_thread(flow.fetch_token, code=code)
        await self._storage.save_credentials(
            user_id=user_id,
            credentials=self._credentials_payload(flow.credentials),
        )
        return True

    def _flow(self, *, code_verifier: str, state: str | None = None) -> Flow:
        return Flow.from_client_config(
            {
                "web": {
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": GOOGLE_OAUTH_EXCHANGE_URL,
                    "redirect_uris": [self._redirect_uri],
                }
            },
            scopes=SCOPES,
            redirect_uri=self._redirect_uri,
            state=state,
            code_verifier=code_verifier,
            autogenerate_code_verifier=False,
        )

    @staticmethod
    def _credentials_payload(credentials: Credentials) -> dict[str, object]:
        payload = json.loads(credentials.to_json())  # type: ignore[no-untyped-call]
        if not isinstance(payload, dict):
            raise TypeError("Google credentials must serialize to an object")
        return payload


def create_oauth_app(*, oauth: GoogleOAuthService) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/oauth/google/callback", response_class=HTMLResponse)
    async def callback(
        code: str | None = Query(default=None),
        state: str | None = Query(default=None),
        error: str | None = Query(default=None),
    ) -> HTMLResponse:
        if error or not code or not state:
            return _oauth_response("<h1>Подключение Google Calendar отменено.</h1>")
        try:
            completed = await oauth.complete(code=code, state=state)
        except Exception:
            logger.exception("Failed to complete Google OAuth")
            return _oauth_response(
                "<h1>Не удалось подключить Google Calendar. Попробуйте снова.</h1>",
                status_code=502,
            )
        if not completed:
            return _oauth_response(
                "<h1>Ссылка подключения недействительна или устарела.</h1>",
                status_code=400,
            )
        return _oauth_response(
            "<h1>Google Calendar подключён. Вернитесь в Telegram.</h1>"
        )

    return app


def _oauth_response(content: str, *, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        content,
        status_code=status_code,
        headers=OAUTH_RESPONSE_HEADERS,
    )
