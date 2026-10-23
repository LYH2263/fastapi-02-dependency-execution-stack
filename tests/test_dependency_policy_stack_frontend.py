"""Frontend route group exit.

``_FrontendRouteGroup`` temporarily swaps ``fastapi_inner_astack`` /
``fastapi_function_astack`` in the ASGI scope while solving its parameterless
dependencies. That is an *exit stack lifecycle* mechanism. These tests prove
that solving before/during/after that swap reads the same request-level
dependency policy stack, while exit stack save/restore semantics are untouched
and no policy frame has to be cleared to swap exit stacks.
"""

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi import (
    Depends,
    FastAPI,
    dependency_policy_frame,
    get_dependency_policy_stack,
    resolve_dependency_override,
)
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class FrameASGIMiddleware:
    def __init__(self, app: ASGIApp, *, overrides: dict | None) -> None:
        self.app = app
        self.overrides = overrides

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            with dependency_policy_frame(overrides=self.overrides):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def test_frontend_group_solve_consumes_pushed_frame(tmp_path: Path) -> None:
    calls: list[str] = []

    def probe_original() -> None:
        calls.append("original")

    def probe_replaced() -> None:
        calls.append("replaced")

    dist = tmp_path / "dist"
    write_file(dist / "index.html", "app")
    app = FastAPI(dependencies=[Depends(probe_original)])
    app.frontend("/", directory=dist)

    response = TestClient(
        FrameASGIMiddleware(app, overrides={probe_original: probe_replaced})
    ).get("/")

    assert response.status_code == 200
    assert response.text == "app"
    assert calls == ["replaced"]

    # The frame is gone after the request; the original dependency runs on a
    # request without the middleware.
    response_plain = TestClient(app).get("/")
    assert response_plain.status_code == 200
    assert calls == ["replaced", "original"]


@pytest.mark.anyio
async def test_frontend_group_swap_keeps_policy_stack_and_restores_exit_stacks(
    tmp_path: Path,
) -> None:
    events: list[tuple[Any, ...]] = []
    scope_holder: dict[str, Any] = {}

    def original_dep() -> None:
        # While the frontend group solves dependencies the exit stacks in the
        # scope have been swapped to fresh ones...
        inner_swapped = (
            scope_holder["scope"]["fastapi_inner_astack"]
            is not scope_holder["sentinel_inner"]
        )
        function_swapped = (
            scope_holder["scope"]["fastapi_function_astack"]
            is not scope_holder["sentinel_function"]
        )
        # ...but the dependency policy stack pushed by the caller is still the
        # visible top, independently of the exit-stack swap.
        events.append(
            (
                len(get_dependency_policy_stack()),
                resolve_dependency_override(original_dep) is replacement_dep,
                inner_swapped,
                function_swapped,
            )
        )

    def replacement_dep() -> None:
        original_dep()

    dist = tmp_path / "dist"
    write_file(dist / "index.html", "app")
    app = FastAPI(dependencies=[Depends(original_dep)])
    app.frontend("/", directory=dist)
    group = app.router._frontend_routes
    assert group is not None

    sentinel_inner = AsyncExitStack()
    sentinel_function = AsyncExitStack()
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "fastapi_inner_astack": sentinel_inner,
        "fastapi_function_astack": sentinel_function,
    }
    scope_holder["scope"] = scope
    scope_holder["sentinel_inner"] = sentinel_inner
    scope_holder["sentinel_function"] = sentinel_function
    messages: list[dict[str, Any]] = []

    async def receive() -> None:
        await anyio.sleep_forever()

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    # An outer policy frame is active *across* the frontend group swapping its
    # exit stacks. No requirement to clear the policy stack first.
    with dependency_policy_frame(overrides={original_dep: replacement_dep}):
        async with sentinel_inner, sentinel_function:
            await group.handle(scope, receive, send)

        # Exit stacks restored exactly as before the group ran...
        assert scope["fastapi_inner_astack"] is sentinel_inner
        assert scope["fastapi_function_astack"] is sentinel_function
        # ...and the outer policy frame is still the visible top (the group
        # never pops the caller's frame back to an empty stack).
        assert len(get_dependency_policy_stack()) == 1
        assert resolve_dependency_override(original_dep) is replacement_dep

    assert get_dependency_policy_stack() == ()
    assert events == [(1, True, True, True)]
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
