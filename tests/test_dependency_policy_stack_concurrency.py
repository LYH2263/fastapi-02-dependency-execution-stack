"""Concurrency exits.

Two requests issued concurrently over the ASGI transport push different
frames (different overrides and different ``cache_partition`` values). Each
request's solving — and streaming — must observe only its own stack top: the
other request's replacement callable must never appear, and push/pop in one
request must not rewrite the top read by the other. TestClient (portal
threads) and a directly-mounted ASGI transport must behave the same.
"""

import threading
from typing import Any

import anyio
import httpx
import pytest
from fastapi import (
    Depends,
    FastAPI,
    get_effective_cache_partition,
    get_effective_dependency_overrides,
)
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send


def shared_dependency() -> str:
    return "original"  # pragma: no cover


def replacement_a() -> str:
    return "a"


def replacement_b() -> str:
    return "b"


class PerPathFrameMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from fastapi import dependency_policy_frame

        if scope["type"] == "http" and scope["path"] in ("/a", "/b"):
            replacement = replacement_a if scope["path"] == "/a" else replacement_b
            with dependency_policy_frame(
                overrides={shared_dependency: replacement},
                cache_partition=scope["path"],
            ):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def build_app(state: dict[str, Any]) -> ASGIApp:
    app = FastAPI()

    async def gated_dependency(request: Request) -> tuple[str, str]:
        who = request.url.path
        other = "/b" if who == "/a" else "/a"

        view = get_effective_dependency_overrides()
        first = view[shared_dependency]
        partition = get_effective_cache_partition()

        state[f"{who}_in"] = True
        while not state.get(f"{other}_in"):
            await anyio.sleep(0)

        # The other request is now solving inside its own frame. Re-reading
        # must still return our own replacement.
        second = get_effective_dependency_overrides()[shared_dependency]
        state[f"{who}_reread"] = first is second

        if who == "/b":
            # Keep request B alive while A streams its response body, then
            # read the view once more.
            while not state.get("a_stream_done"):
                await anyio.sleep(0)
            third = get_effective_dependency_overrides()[shared_dependency]
            state["b_third"] = third is replacement_b

        return first(), str(partition)

    @app.get("/a")
    async def stream_endpoint(
        data: tuple[str, str] = Depends(gated_dependency),
    ) -> StreamingResponse:
        value, partition = data

        async def generate() -> Any:
            for index in range(3):
                await anyio.sleep(0)
                own_view = get_effective_dependency_overrides().get(shared_dependency)
                state.setdefault("a_stream_checks", []).append(
                    own_view is replacement_a
                )
                state["a_stream_partition"] = get_effective_cache_partition() == "/a"
                yield f"{value}-{index}|".encode()
            state["a_stream_done"] = True

        return StreamingResponse(generate())

    @app.get("/b")
    async def plain_endpoint(
        data: tuple[str, str] = Depends(gated_dependency),
    ) -> dict[str, Any]:
        value, partition = data
        return {
            "value": value,
            "partition": partition,
            "reread": state["/b_reread"],
            "third": state["b_third"],
        }

    return PerPathFrameMiddleware(app)


def _assert_isolated_state(state: dict[str, Any], *, b_body: Any) -> None:
    # Neither request ever re-read the other one's replacement callable.
    assert state["/a_reread"] is True
    assert state["/b_reread"] is True
    assert state["b_third"] is True
    # A kept observing its own frame while streaming, B stayed alive meanwhile.
    assert state["a_stream_checks"] == [True, True, True]
    assert state["a_stream_partition"] is True
    assert b_body["value"] == "b"
    assert b_body["partition"] == "/b"
    assert b_body["reread"] is True
    assert b_body["third"] is True


@pytest.mark.anyio
async def test_concurrent_asgi_requests_have_independent_policy_frames() -> None:
    state: dict[str, Any] = {}
    app = build_app(state)

    transport = httpx.ASGITransport(app=app)
    results: dict[str, Any] = {}

    async def get_a() -> None:
        results["a"] = await client.get("/a")

    async def get_b() -> None:
        results["b"] = await client.get("/b")

    async with httpx.AsyncClient(
        transport=transport, base_url="http://policy-test"
    ) as client:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(get_a)
            task_group.start_soon(get_b)

    response_a = results["a"]
    response_b = results["b"]

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_a.text.split("|")[:3] == ["a-0", "a-1", "a-2"]
    _assert_isolated_state(state, b_body=response_b.json())


def test_concurrent_testclient_requests_have_independent_policy_frames() -> None:
    state: dict[str, Any] = {}
    app = build_app(state)

    bodies: dict[str, Any] = {}

    def hit_a() -> None:
        bodies["a"] = TestClient(app).get("/a").text

    def hit_b() -> None:
        bodies["b"] = TestClient(app).get("/b").json()

    thread_a = threading.Thread(target=hit_a)
    thread_b = threading.Thread(target=hit_b)
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=20)
    thread_b.join(timeout=20)
    assert not thread_a.is_alive() and not thread_b.is_alive()

    assert bodies["a"].split("|")[:3] == ["a-0", "a-1", "a-2"]
    _assert_isolated_state(state, b_body=bodies["b"])
