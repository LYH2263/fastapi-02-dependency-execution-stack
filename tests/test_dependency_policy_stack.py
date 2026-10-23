"""Request-level dependency execution policy stack.

These tests cover the stack itself and the HTTP path-operation exit of
``solve_dependencies``. The other exits (WebSocket, frontend route groups,
OpenAPI generation, background tasks, concurrency) live in dedicated
``test_dependency_policy_stack_*.py`` modules.
"""

from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Annotated, Any

import pytest
from fastapi import (
    DependencyPolicyFrame,
    Depends,
    FastAPI,
    Security,
    dependency_policy_frame,
    get_dependency_policy_stack,
    get_effective_cache_partition,
    get_effective_dependency_overrides,
    get_effective_extra_oauth_scopes,
    pop_dependency_policy,
    push_dependency_policy,
    resolve_dependency_override,
)
from fastapi.dependencies.utils import get_dependant, solve_dependencies
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.oauth2 import SecurityScopes
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

# ---------------------------------------------------------------------------
# Minimal ASGI middleware used to push a frame around a whole request.
# ---------------------------------------------------------------------------


class FrameASGIMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        overrides: dict[Callable[..., Any], Callable[..., Any]] | None = None,
        cache_partition: Any = None,
        extra_oauth_scopes: tuple[str, ...] = (),
        path: str | None = None,
        scope_types: tuple[str, ...] = ("http",),
    ) -> None:
        self.app = app
        self.overrides = overrides
        self.cache_partition = cache_partition
        self.extra_oauth_scopes = extra_oauth_scopes
        self.path = path
        self.scope_types = scope_types

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        active = scope["type"] in self.scope_types and (
            self.path is None or scope.get("path") == self.path
        )
        if active:
            with dependency_policy_frame(
                overrides=self.overrides,
                cache_partition=self.cache_partition,
                extra_oauth_scopes=self.extra_oauth_scopes,
            ):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Stack mechanics
# ---------------------------------------------------------------------------


def test_empty_stack_matches_application_only_view() -> None:
    assert get_dependency_policy_stack() == ()
    assert get_effective_dependency_overrides() == {}
    assert get_effective_cache_partition() is None
    assert get_effective_extra_oauth_scopes() == ()

    def original() -> str:
        return "original"

    assert resolve_dependency_override(original) is original


def test_inner_frame_wins_outer_frame_then_app_then_original() -> None:
    def original() -> str:
        return "original"

    def app_replacement() -> str:
        return "app"

    def outer_replacement() -> str:
        return "outer"

    def inner_replacement() -> str:
        return "inner"

    class Provider:
        dependency_overrides = {original: app_replacement}

    provider = Provider()

    # No frames: application-level overrides apply.
    assert resolve_dependency_override(original, provider) is app_replacement

    with dependency_policy_frame(overrides={original: outer_replacement}):
        assert resolve_dependency_override(original, provider) is outer_replacement
        assert (
            get_effective_dependency_overrides(provider)[original] is outer_replacement
        )
        with dependency_policy_frame(overrides={original: inner_replacement}):
            assert resolve_dependency_override(original, provider) is inner_replacement
            assert len(get_dependency_policy_stack()) == 2
        # Popping the inner frame reveals the outer frame again, never an
        # empty stack or a process-wide singleton.
        assert resolve_dependency_override(original, provider) is outer_replacement
        assert len(get_dependency_policy_stack()) == 1

    # After popping the outer frame only the application mapping remains.
    assert resolve_dependency_override(original, provider) is app_replacement
    assert get_dependency_policy_stack() == ()


def test_merged_overrides_keep_unrelated_entries_from_all_layers() -> None:
    def dep_a() -> str:
        return "a"

    def dep_b() -> str:
        return "b"

    def dep_c() -> str:
        return "c"

    def app_b() -> str:
        return "app-b"

    def frame_a() -> str:
        return "frame-a"

    def frame_b() -> str:
        return "frame-b"

    class Provider:
        dependency_overrides = {dep_b: app_b, dep_c: dep_c}

    with dependency_policy_frame(overrides={dep_a: frame_a, dep_b: frame_b}):
        merged = get_effective_dependency_overrides(Provider())
        assert merged[dep_a] is frame_a
        assert merged[dep_b] is frame_b  # frame beats app
        assert merged[dep_c] is dep_c  # app-level entry untouched


def test_context_manager_pops_on_exception() -> None:
    def original() -> None:
        pass

    def replacement() -> None:
        pass

    with pytest.raises(RuntimeError):
        with dependency_policy_frame(overrides={original: replacement}):
            assert len(get_dependency_policy_stack()) == 1
            raise RuntimeError("boom")

    assert get_dependency_policy_stack() == ()


def test_push_and_pop_token_api_restores_previous_stack() -> None:
    def original() -> None:
        pass

    def outer() -> None:
        pass

    def inner() -> None:
        pass

    outer_token = push_dependency_policy(
        DependencyPolicyFrame(overrides={original: outer})
    )
    inner_token = push_dependency_policy(
        DependencyPolicyFrame(overrides={original: inner})
    )
    assert resolve_dependency_override(original) is inner
    pop_dependency_policy(inner_token)
    assert resolve_dependency_override(original) is outer
    pop_dependency_policy(outer_token)
    assert get_dependency_policy_stack() == ()


def test_extra_scopes_merge_outer_to_inner_deduplicated() -> None:
    with dependency_policy_frame(extra_oauth_scopes=["a", "b"]):
        assert get_effective_extra_oauth_scopes() == ("a", "b")
        with dependency_policy_frame(extra_oauth_scopes=["b", "c"]):
            assert get_effective_extra_oauth_scopes() == ("a", "b", "c")
            assert get_effective_cache_partition() is None
        assert get_effective_extra_oauth_scopes() == ("a", "b")
    assert get_effective_extra_oauth_scopes() == ()


def test_cache_partition_comes_from_topmost_frame() -> None:
    with dependency_policy_frame(cache_partition="outer"):
        assert get_effective_cache_partition() == "outer"
        with dependency_policy_frame(cache_partition="inner"):
            assert get_effective_cache_partition() == "inner"
        assert get_effective_cache_partition() == "outer"
    assert get_effective_cache_partition() is None


# ---------------------------------------------------------------------------
# HTTP exit: solve_dependencies called from a path operation
# ---------------------------------------------------------------------------


def test_http_solve_uses_stack_top_override_and_pops_after_request() -> None:
    def original_dep() -> str:
        return "original"

    def frame_dep() -> str:
        return "frame"

    app = FastAPI()

    @app.get("/")
    def root(value: str = Depends(original_dep)) -> dict[str, str]:
        return {"value": value}

    wrapped = FrameASGIMiddleware(app, overrides={original_dep: frame_dep})
    client = TestClient(wrapped)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"value": "frame"}

    # A request that does not push a frame sees the original callable, proving
    # the previous request did not leave any frame behind.
    plain_client = TestClient(app)
    assert plain_client.get("/").json() == {"value": "original"}


def test_http_frame_override_takes_precedence_over_app_overrides() -> None:
    def original_dep() -> str:
        return "original"

    def app_dep() -> str:
        return "app"

    def frame_dep() -> str:
        return "frame"

    app = FastAPI()
    app.dependency_overrides[original_dep] = app_dep

    @app.get("/")
    def root(value: str = Depends(original_dep)) -> dict[str, str]:
        return {"value": value}

    # No frame: the application-level mapping is used as today.
    assert TestClient(app).get("/").json() == {"value": "app"}

    # Frame active: it wins over the application mapping; the application
    # recovery order is independent (the mapping stays usable afterwards).
    wrapped = FrameASGIMiddleware(app, overrides={original_dep: frame_dep})
    assert TestClient(wrapped).get("/").json() == {"value": "frame"}
    assert TestClient(app).get("/").json() == {"value": "app"}


def test_http_extra_oauth_scopes_append_on_top_of_declared_scopes() -> None:
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

    def get_scopes(
        security_scopes: SecurityScopes,
        token: Annotated[str, Depends(oauth2_scheme)],
    ) -> list[str]:
        return security_scopes.scopes

    app = FastAPI()

    @app.get("/")
    def root(
        scopes: Annotated[list[str], Security(get_scopes, scopes=["me"])],
    ) -> dict[str, list[str]]:
        return {"scopes": scopes}

    # No frame: only the statically declared scopes are injected.
    plain = TestClient(app).get("/", headers={"authorization": "Bearer token"})
    assert plain.json() == {"scopes": ["me"]}

    # A frame appends temporary scopes to the injected SecurityScopes.
    wrapped = FrameASGIMiddleware(app, extra_oauth_scopes=("admin",))
    framed = TestClient(wrapped).get("/", headers={"authorization": "Bearer token"})
    assert framed.json() == {"scopes": ["me", "admin"]}

    # The extra scope is gone on the next request.
    plain_again = TestClient(app).get("/", headers={"authorization": "Bearer token"})
    assert plain_again.json() == {"scopes": ["me"]}


def test_http_use_cache_still_shares_within_one_framed_tree() -> None:
    calls: list[str] = []

    def shared_dep() -> str:
        calls.append("called")
        return "value"

    app = FastAPI()

    @app.get("/")
    def root(
        first: str = Depends(shared_dep),
        second: str = Depends(shared_dep),
    ) -> dict[str, str]:
        return {"first": first, "second": second}

    wrapped = FrameASGIMiddleware(app, cache_partition="partition")
    response = TestClient(wrapped).get("/")

    assert response.json() == {"first": "value", "second": "value"}
    assert calls == ["called"]


def test_http_use_cache_false_still_runs_again_under_frame() -> None:
    calls = 0

    def uncached_dep() -> int:
        nonlocal calls
        calls += 1
        return calls

    app = FastAPI()

    @app.get("/")
    def root(
        first: int = Depends(uncached_dep, use_cache=False),
        second: int = Depends(uncached_dep, use_cache=False),
    ) -> dict[str, int]:
        return {"first": first, "second": second}

    wrapped = FrameASGIMiddleware(app, cache_partition="partition")
    response = TestClient(wrapped).get("/")

    assert response.json() == {"first": 1, "second": 2}
    assert calls == 2


# ---------------------------------------------------------------------------
# Direct solving: cache partition and extra scopes are a new cache dimension.
# ---------------------------------------------------------------------------


async def _solve_tree(
    app: FastAPI,
    endpoint: Callable[..., Any],
    *,
    dependency_cache: dict[Any, Any] | None = None,
) -> Any:
    async with AsyncExitStack() as inner_stack, AsyncExitStack() as function_stack:
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
            "fastapi_inner_astack": inner_stack,
            "fastapi_function_astack": function_stack,
        }
        request = Request(scope)
        dependant = get_dependant(path="/", call=endpoint)
        solved = await solve_dependencies(
            request=request,
            dependant=dependant,
            dependency_overrides_provider=app,
            dependency_cache=dependency_cache,
            async_exit_stack=inner_stack,
            embed_body_fields=False,
        )
        assert not solved.errors
        return solved


def _frame(
    *,
    cache_partition: Any = None,
    extra_oauth_scopes: tuple[str, ...] = (),
) -> Any:
    return dependency_policy_frame(
        cache_partition=cache_partition,
        extra_oauth_scopes=extra_oauth_scopes,
    )


@pytest.mark.anyio
async def test_different_cache_partitions_never_share_cached_value() -> None:
    calls: list[int] = []

    def counted_dep() -> int:
        calls.append(1)
        return len(calls)

    def endpoint(value: int = Depends(counted_dep)) -> int:
        return value  # pragma: no cover

    app = FastAPI()
    cache: dict[Any, Any] = {}

    with _frame(cache_partition="a"):
        first = await _solve_tree(app, endpoint, dependency_cache=cache)
    with _frame(cache_partition="b"):
        second = await _solve_tree(app, endpoint, dependency_cache=cache)

    assert first.values["value"] == 1
    assert second.values["value"] == 2
    assert len(calls) == 2


@pytest.mark.anyio
async def test_same_cache_partition_reuses_cached_value() -> None:
    calls: list[int] = []

    def counted_dep() -> int:
        calls.append(1)
        return len(calls)

    def endpoint(value: int = Depends(counted_dep)) -> int:
        return value  # pragma: no cover

    app = FastAPI()
    cache: dict[Any, Any] = {}

    with _frame(cache_partition="a"):
        first = await _solve_tree(app, endpoint, dependency_cache=cache)
    with _frame(cache_partition="a"):
        second = await _solve_tree(app, endpoint, dependency_cache=cache)

    assert first.values["value"] == 1
    assert second.values["value"] == 1
    assert len(calls) == 1


@pytest.mark.anyio
async def test_default_partition_does_not_share_with_named_partition() -> None:
    calls: list[int] = []

    def counted_dep() -> int:
        calls.append(1)
        return len(calls)

    def endpoint(value: int = Depends(counted_dep)) -> int:
        return value  # pragma: no cover

    app = FastAPI()
    cache: dict[Any, Any] = {}

    with _frame(cache_partition="named"):
        named = await _solve_tree(app, endpoint, dependency_cache=cache)
    # No frame at all: historical default cache keys must not hit the entry
    # stored under the named partition.
    empty = await _solve_tree(app, endpoint, dependency_cache=cache)

    assert named.values["value"] == 1
    assert empty.values["value"] == 2
    assert len(calls) == 2


@pytest.mark.anyio
async def test_different_extra_oauth_scopes_do_not_share_security_cache() -> None:
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)
    calls: list[list[str]] = []

    def security_dep(
        security_scopes: SecurityScopes,
        token: Annotated[str, Depends(oauth2_scheme)],
    ) -> list[str]:
        calls.append(list(security_scopes.scopes))
        return list(security_scopes.scopes)

    def endpoint(scopes: list[str] = Depends(security_dep)) -> list[str]:
        return scopes  # pragma: no cover

    app = FastAPI()
    cache: dict[Any, Any] = {}

    with _frame(extra_oauth_scopes=("scope-a",)):
        first = await _solve_tree(app, endpoint, dependency_cache=cache)
    with _frame(extra_oauth_scopes=("scope-b",)):
        second = await _solve_tree(app, endpoint, dependency_cache=cache)

    assert first.values["scopes"] == ["scope-a"]
    assert second.values["scopes"] == ["scope-b"]
    # Different oauth scope combinations must never share a cache entry, even
    # when the callable is the same.
    assert calls == [["scope-a"], ["scope-b"]]

    with _frame(extra_oauth_scopes=("scope-a",)):
        reused = await _solve_tree(app, endpoint, dependency_cache=cache)
    assert reused.values["scopes"] == ["scope-a"]
    assert len(calls) == 2


@pytest.mark.anyio
async def test_frame_does_not_cancel_static_scope_cache_separation() -> None:
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)
    calls: list[list[str]] = []

    def make_endpoint(static_scope: str) -> Callable[..., Any]:
        def security_dep(
            security_scopes: SecurityScopes,
            token: Annotated[str, Depends(oauth2_scheme)],
        ) -> list[str]:
            calls.append(list(security_scopes.scopes))
            return list(security_scopes.scopes)

        def endpoint(
            scopes: Annotated[list[str], Security(security_dep, scopes=[static_scope])],
        ) -> list[str]:
            return scopes  # pragma: no cover

        return endpoint

    app = FastAPI()
    cache: dict[Any, Any] = {}

    # Same frame, same partition, but two *statically* different scope
    # combinations: the existing non-sharing rule still applies.
    with _frame(cache_partition="p", extra_oauth_scopes=("extra",)):
        first = await _solve_tree(app, make_endpoint("read"), dependency_cache=cache)
        second = await _solve_tree(app, make_endpoint("write"), dependency_cache=cache)

    assert first.values["scopes"] == ["read", "extra"]
    assert second.values["scopes"] == ["write", "extra"]
    assert calls == [["read", "extra"], ["write", "extra"]]


@pytest.mark.anyio
async def test_frame_overrides_apply_in_direct_solve_without_provider() -> None:
    def original_dep() -> str:
        return "original"

    def replacement_dep() -> str:
        return "replaced"

    def endpoint(value: str = Depends(original_dep)) -> str:
        return value  # pragma: no cover

    app = FastAPI()

    with dependency_policy_frame(overrides={original_dep: replacement_dep}):
        solved = await _solve_tree(app, endpoint)

    assert solved.values["value"] == "replaced"


def test_async_callable_replacement_runs() -> None:
    async def original_dep() -> str:
        return "original"

    async def replacement_dep() -> str:
        return "replaced"

    app = FastAPI()

    @app.get("/")
    async def root(value: str = Depends(original_dep)) -> dict[str, str]:
        return {"value": value}

    wrapped = FrameASGIMiddleware(app, overrides={original_dep: replacement_dep})
    response = TestClient(wrapped).get("/")
    assert response.json() == {"value": "replaced"}


def test_pushed_frame_is_a_policy_frame_instance() -> None:
    with dependency_policy_frame() as frame:
        assert isinstance(frame, DependencyPolicyFrame)
    assert get_dependency_policy_stack() == ()
