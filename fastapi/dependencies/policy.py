"""Request-level dependency execution policy stack.

This stack is intentionally independent from:

* ``app.dependency_overrides`` — the application-level override mapping, which
  keeps working exactly as before; frames pushed here take precedence over it.
* ``Depends(scope="function" | "request")`` and the
  ``fastapi_inner_astack`` / ``fastapi_function_astack`` selection — those
  control *which* ``AsyncExitStack`` owns a dependency's teardown, not how
  dependencies are overridden, cached or scoped.
* The lifespan and any routing machinery — no constructor of ``FastAPI``,
  ``APIRouter`` or ``Request`` gains a parameter.

The stack is stored in a :class:`contextvars.ContextVar`. Every HTTP request
and every accepted WebSocket connection runs in its own context, so concurrent
requests can push frames without ever observing each other. The empty stack
has exactly the same observable behavior as not using this module at all.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

PolicyStack = tuple["DependencyPolicyFrame", ...]

#: Sentinel for "no cache partition selected" — i.e. the historical default.
DEFAULT_CACHE_PARTITION: Any = None


_policy_stack_var: ContextVar[PolicyStack] = ContextVar(
    "fastapi_dependency_policy_stack", default=()
)


@dataclass(frozen=True, slots=True)
class DependencyPolicyFrame:
    """One layer of the request-level dependency execution policy stack.

    ``overrides`` maps an original dependency callable to the callable that
    should replace it while the frame is active. Inner frames win over outer
    frames, and every frame wins over ``app.dependency_overrides``.

    ``cache_partition`` adds a partition dimension to the dependency cache:
    a value solved in one partition is never reused in another one, even when
    the callable and OAuth scopes are identical. ``None`` is the historical
    default partition.

    ``extra_oauth_scopes`` are appended (deduplicated, order preserving) to the
    scopes a ``dependant`` already declares whenever ``SecurityScopes`` is
    injected while the frame is active.
    """

    overrides: Mapping[Callable[..., Any], Callable[..., Any]] = field(
        default_factory=dict
    )
    cache_partition: Any = DEFAULT_CACHE_PARTITION
    extra_oauth_scopes: tuple[str, ...] = ()


def _coerce_overrides(
    overrides: Mapping[Callable[..., Any], Callable[..., Any]] | None,
) -> dict[Callable[..., Any], Callable[..., Any]]:
    if overrides is None:
        return {}
    return dict(overrides)


def push_dependency_policy(frame: DependencyPolicyFrame) -> Token[PolicyStack]:
    """Push a frame onto the current context's policy stack.

    Returns a :class:`contextvars.Token` that must be handed back to
    :func:`pop_dependency_policy`, in LIFO order. Prefer
    :func:`dependency_policy_frame` for automatic cleanup on exception paths.
    """
    return _policy_stack_var.set(_policy_stack_var.get() + (frame,))


def pop_dependency_policy(token: Token[PolicyStack]) -> None:
    """Pop the frame pushed with ``token``, restoring the previous stack.

    After popping an inner frame the outer frame's policy is visible again —
    the stack never collapses to a process-wide singleton. ``Token`` semantics
    enforce strict LIFO usage.
    """
    _policy_stack_var.reset(token)


@contextmanager
def dependency_policy_frame(
    *,
    overrides: Mapping[Callable[..., Any], Callable[..., Any]] | None = None,
    cache_partition: Any = DEFAULT_CACHE_PARTITION,
    extra_oauth_scopes: Sequence[str] = (),
) -> Iterator[DependencyPolicyFrame]:
    """Context manager that pushes a frame and always pops it again.

    The frame is removed on normal exit *and* on exception propagation, so
    frames cannot leak onto subsequent solves in the same request context.
    """
    frame = DependencyPolicyFrame(
        overrides=_coerce_overrides(overrides),
        cache_partition=cache_partition,
        extra_oauth_scopes=tuple(extra_oauth_scopes),
    )
    token = push_dependency_policy(frame)
    try:
        yield frame
    finally:
        pop_dependency_policy(token)


def get_dependency_policy_stack() -> PolicyStack:
    """Return the frames currently pushed, outermost first."""
    return _policy_stack_var.get()


def get_effective_cache_partition() -> Any:
    """Return the partition of the topmost frame, or the default partition."""
    stack = _policy_stack_var.get()
    if not stack:
        return DEFAULT_CACHE_PARTITION
    return stack[-1].cache_partition


def get_effective_extra_oauth_scopes() -> tuple[str, ...]:
    """Merge ``extra_oauth_scopes`` across frames, outermost first, deduplicated."""
    scopes: list[str] = []
    for frame in _policy_stack_var.get():
        for scope in frame.extra_oauth_scopes:
            if scope not in scopes:
                scopes.append(scope)
    return tuple(scopes)


def get_effective_dependency_overrides(
    dependency_overrides_provider: Any | None = None,
) -> Mapping[Callable[..., Any], Callable[..., Any]]:
    """Effective override view: application overrides first, then frames.

    The merged mapping applies outer frames first and inner frames last, so
    inner frames win. It never mutates ``app.dependency_overrides``.
    """
    stack = _policy_stack_var.get()
    app_overrides = (
        getattr(dependency_overrides_provider, "dependency_overrides", None)
        if dependency_overrides_provider is not None
        else None
    )
    if not stack:
        return app_overrides or {}
    merged: dict[Callable[..., Any], Callable[..., Any]] = {}
    if app_overrides:
        merged.update(app_overrides)
    for frame in stack:
        merged.update(frame.overrides)
    return merged


def resolve_dependency_override(
    call: Callable[..., Any],
    dependency_overrides_provider: Any | None = None,
) -> Callable[..., Any]:
    """Resolve one dependency callable against the effective view.

    Lookup order: inner stack frames first, then outer frames, then the
    application-level ``dependency_overrides`` mapping, finally the original
    callable unchanged.
    """
    for frame in reversed(_policy_stack_var.get()):
        if call in frame.overrides:
            return frame.overrides[call]
    app_overrides = (
        getattr(dependency_overrides_provider, "dependency_overrides", None)
        if dependency_overrides_provider is not None
        else None
    )
    if app_overrides:
        return app_overrides.get(call, call)
    return call


@dataclass(frozen=True, slots=True)
class _DependencyPolicySnapshot:
    """Frozen view captured once for a whole ``solve_dependencies`` tree."""

    overrides: Mapping[Callable[..., Any], Callable[..., Any]]
    cache_partition: Any
    extra_oauth_scopes: tuple[str, ...]


def capture_dependency_policy(
    dependency_overrides_provider: Any | None = None,
) -> _DependencyPolicySnapshot | None:
    """Snapshot the current stack.

    Returns ``None`` when no frame is pushed, which callers treat as the exact
    historical code path (application-level overrides and default cache only).
    """
    if not _policy_stack_var.get():
        return None
    return _DependencyPolicySnapshot(
        overrides=get_effective_dependency_overrides(
            dependency_overrides_provider=dependency_overrides_provider
        ),
        cache_partition=get_effective_cache_partition(),
        extra_oauth_scopes=get_effective_extra_oauth_scopes(),
    )


def policy_partitions_cache_key(
    cache_key: Any,
    *,
    cache_partition: Any,
    extra_oauth_scopes: tuple[str, ...],
) -> Any:
    """Add the partition/extra-scope dimension to a static cache key.

    Only applied while the stack is active and only when the partition or
    extra scopes actually differ from the historical defaults, so empty-stack
    keys remain byte-for-byte the 3-tuples used today.
    """
    if cache_partition is DEFAULT_CACHE_PARTITION and not extra_oauth_scopes:
        return cache_key
    return (cache_key, cache_partition, extra_oauth_scopes)


@contextmanager
def _empty_dependency_policy_stack() -> Iterator[None]:
    """Run detached from any frames leaked in the surrounding context.

    Used for work that executes after the request is done (background tasks):
    such work must observe an empty request policy stack while the
    application-level overrides stay in effect.
    """
    token = _policy_stack_var.set(())
    try:
        yield
    finally:
        _policy_stack_var.reset(token)
