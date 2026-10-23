"""WebSocket exit: dependency solving for an accepted connection must consume
the same request-level policy stack top as HTTP solving, without going through
any HTTP response encoding/media type logic.
"""

from typing import Annotated

import pytest
from fastapi import Depends, FastAPI, Security, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.oauth2 import SecurityScopes
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send


class FrameASGIMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        overrides: dict | None = None,
        extra_oauth_scopes: tuple[str, ...] = (),
        cache_partition: object | None = None,
    ) -> None:
        self.app = app
        self.overrides = overrides
        self.extra_oauth_scopes = extra_oauth_scopes
        self.cache_partition = cache_partition

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from fastapi import dependency_policy_frame

        if scope["type"] == "websocket":
            with dependency_policy_frame(
                overrides=self.overrides,
                extra_oauth_scopes=self.extra_oauth_scopes,
                cache_partition=self.cache_partition,
            ):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def test_websocket_solve_uses_stack_top_override() -> None:
    def connection_label() -> str:
        return "original"

    def framed_connection_label() -> str:
        return "framed"

    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(
        websocket: WebSocket, label: str = Depends(connection_label)
    ) -> None:
        await websocket.accept()
        await websocket.send_json({"label": label})
        await websocket.close()

    # No frame: original dependency.
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json() == {"label": "original"}

    # A frame bound to the accepted WebSocket connection replaces it.
    framed_app = FrameASGIMiddleware(
        app, overrides={connection_label: framed_connection_label}
    )
    with TestClient(framed_app) as client:
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json() == {"label": "framed"}

    # The frame did not leak onto a subsequent connection.
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json() == {"label": "original"}


def test_websocket_solve_appends_extra_oauth_scopes() -> None:
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

    def get_scopes(
        security_scopes: SecurityScopes,
        token: Annotated[str, Depends(oauth2_scheme)],
    ) -> list[str]:
        return security_scopes.scopes

    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(
        websocket: WebSocket,
        scopes: Annotated[list[str], Security(get_scopes, scopes=["connect"])],
    ) -> None:
        await websocket.accept()
        await websocket.send_json({"scopes": scopes})
        await websocket.close()

    framed_app = FrameASGIMiddleware(app, extra_oauth_scopes=("admin",))
    with TestClient(framed_app) as client:
        with client.websocket_connect(
            "/ws", headers={"authorization": "Bearer token"}
        ) as websocket:
            assert websocket.receive_json() == {"scopes": ["connect", "admin"]}


def test_websocket_frame_does_not_mutate_static_dependant_graph() -> None:
    # Validation errors from WebSocket solving keep flowing through the
    # WebSocket-specific path (WebSocketRequestValidationError -> close 1008):
    # a frame cannot turn the connection into HTTP response handling.
    def require_token(token: str) -> str:
        return token

    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(
        websocket: WebSocket, token: str = Depends(require_token)
    ) -> None:
        await websocket.accept()
        await websocket.send_text(token)
        await websocket.close()

    framed_app = FrameASGIMiddleware(app, cache_partition="ws-partition")
    with TestClient(framed_app) as client:
        # Missing required query parameter: the WebSocket-specific validation
        # path closes the handshake (1008) instead of producing an HTTP body.
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws"):
                pass  # pragma: no cover
        assert exc_info.value.code == 1008
