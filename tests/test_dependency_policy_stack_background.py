"""Background tasks exit.

``BackgroundTasks`` run after the response has been handed to the client.
They must observe an empty request-level policy stack: a task that reads the
merged override view (or triggers dependency solving again) must never see a
frame the request forgot to pop. Application-level ``dependency_overrides``
keep working inside tasks.
"""

from typing import Any

from fastapi import (
    BackgroundTasks,
    DependencyPolicyFrame,
    FastAPI,
    dependency_policy_frame,
    get_dependency_policy_stack,
    get_effective_cache_partition,
    get_effective_extra_oauth_scopes,
    push_dependency_policy,
    resolve_dependency_override,
)
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send


def dep_a() -> str:
    return "a-original"


def dep_b() -> str:
    return "b-original"


def frame_a() -> str:
    return "a-frame"


def app_b() -> str:
    return "b-app"


class FrameASGIMiddleware:
    def __init__(self, app: ASGIApp, *, overrides: dict) -> None:
        self.app = app
        self.overrides = overrides

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            with dependency_policy_frame(
                overrides=self.overrides,
                cache_partition="request-partition",
                extra_oauth_scopes=("task:scope",),
            ):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def _build_app(events: list[tuple[Any, ...]]) -> FastAPI:
    app = FastAPI()
    app.dependency_overrides[dep_b] = app_b

    def record(label: str) -> None:
        events.append(
            (
                label,
                len(get_dependency_policy_stack()),
                resolve_dependency_override(dep_a, app),
                resolve_dependency_override(dep_b, app),
                get_effective_cache_partition(),
                get_effective_extra_oauth_scopes(),
            )
        )

    @app.get("/")
    def root(background_tasks: BackgroundTasks) -> dict[str, bool]:
        # While handling the request the frame is the visible top.
        assert len(get_dependency_policy_stack()) == 1
        assert resolve_dependency_override(dep_a, app) is frame_a
        background_tasks.add_task(record, "async-task")
        background_tasks.add_task(record, "sync-task")
        return {"ok": True}

    return app


def test_background_tasks_see_empty_stack_but_keep_app_overrides() -> None:
    events: list[tuple[Any, ...]] = []
    app = _build_app(events)
    wrapped = FrameASGIMiddleware(app, overrides={dep_a: frame_a})

    response = TestClient(wrapped).get("/")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert events == [
        (
            "async-task",
            0,  # stack depth
            dep_a,  # frame override gone -> original callable
            app_b,  # application-level overrides still apply
            None,  # default partition
            (),  # no extra scopes
        ),
        (
            "sync-task",
            0,
            dep_a,
            app_b,
            None,
            (),
        ),
    ]
    # After the full response cycle the request task context is gone too.
    assert get_dependency_policy_stack() == ()


class LeakyFrameMiddleware:
    """Pushes a frame and "forgets" to pop it, on purpose."""

    def __init__(self, app: ASGIApp, *, overrides: dict) -> None:
        self.app = app
        self.overrides = overrides

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            push_dependency_policy(DependencyPolicyFrame(overrides=self.overrides))
            await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def test_background_tasks_cannot_read_a_leaked_frame() -> None:
    seen: list[object] = []

    app = FastAPI()

    def inspect() -> None:
        seen.append(
            (
                len(get_dependency_policy_stack()),
                resolve_dependency_override(dep_a, app),
            )
        )

    @app.get("/")
    def root(background_tasks: BackgroundTasks) -> dict[str, bool]:
        background_tasks.add_task(inspect)
        return {"ok": True}

    response = TestClient(LeakyFrameMiddleware(app, overrides={dep_a: frame_a})).get(
        "/"
    )

    assert response.status_code == 200
    assert seen == [(0, dep_a)]
