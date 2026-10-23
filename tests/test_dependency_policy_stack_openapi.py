"""OpenAPI generation exit.

Generating the OpenAPI schema (and its security schemes) traverses only the
static ``Dependant`` graph and route declarations. Request-time policy stack
frames — temporary overrides and ``extra_oauth_scopes`` — must never be written
into the document.
"""

from typing import Annotated

from fastapi import Depends, FastAPI, Security, dependency_policy_frame
from fastapi.security import OAuth2PasswordBearer
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


def build_app() -> FastAPI:
    def get_scopes_keeper(
        token: Annotated[str, Depends(oauth2_scheme)],
    ) -> str:
        return token

    app = FastAPI()

    @app.get(
        "/items/",
        dependencies=[Security(get_scopes_keeper, scopes=["items:read"])],
    )
    def read_items() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_openapi_ignores_frames_when_generated_inside_one() -> None:
    app_without = build_app()
    app_with = build_app()

    def temporary_dependency() -> str:
        return "temp"  # pragma: no cover

    spec_before = app_without.openapi()

    with dependency_policy_frame(
        overrides={oauth2_scheme: temporary_dependency},
        cache_partition="documentation",
        extra_oauth_scopes=["items:admin", "ghost:scope"],
    ):
        spec_inside = app_with.openapi()

    assert spec_inside == spec_before


def test_openapi_security_schemes_keep_only_static_scopes() -> None:
    app = build_app()

    def temporary_dependency() -> str:
        return "temp"  # pragma: no cover

    with dependency_policy_frame(
        overrides={oauth2_scheme: temporary_dependency},
        extra_oauth_scopes=["items:admin"],
    ):
        spec = app.openapi()

    operation = spec["paths"]["/items/"]["get"]
    assert operation["security"] == [{"OAuth2PasswordBearer": ["items:read"]}]
    assert "items:admin" not in str(spec["components"]["securitySchemes"])
    assert spec["components"]["securitySchemes"]["OAuth2PasswordBearer"] == {
        "flows": {
            "password": {
                "scopes": {},
                "tokenUrl": "/token",
            }
        },
        "type": "oauth2",
    }


class FrameASGIMiddleware:
    def __init__(self, app: ASGIApp, *, overrides: dict) -> None:
        self.app = app
        self.overrides = overrides

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            with dependency_policy_frame(
                overrides=self.overrides, extra_oauth_scopes=("items:admin",)
            ):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def test_openapi_endpoint_served_under_frame_matches_plain_app() -> None:
    plain = build_app()
    expected = TestClient(plain).get("/openapi.json").json()

    def temporary_dependency() -> str:
        return "temp"  # pragma: no cover

    framed = FrameASGIMiddleware(
        build_app(), overrides={oauth2_scheme: temporary_dependency}
    )
    served = TestClient(framed).get("/openapi.json").json()

    assert served == expected
    assert "items:admin" not in str(served)
