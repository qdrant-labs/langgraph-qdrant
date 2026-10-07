"""Plans: store logic written once as generators that yield requests.

`QdrantStore` and `AsyncQdrantStore` only execute the requests, synchronously
or with asyncio.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException

T = TypeVar("T")

TRANSPORT_ATTEMPTS = 3


@dataclass(frozen=True)
class Call:
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class EmbedDocuments:
    texts: list[str]


@dataclass(frozen=True)
class EmbedQueries:
    queries: list[str]


@dataclass(frozen=True)
class Gather:
    """Independent plans; the async store runs them concurrently."""

    plans: list[Plan[Any]]


Request = Call | EmbedDocuments | EmbedQueries | Gather
Plan = Generator[Request, Any, T]


def call(method: str, *args: Any, **kwargs: Any) -> Plan[Any]:
    # Every request is idempotent, so transient transport failures are retried.
    for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
        try:
            return (yield Call(method, args, kwargs))
        except ResponseHandlingException as exc:
            if attempt == TRANSPORT_ATTEMPTS or not is_transient(exc):
                raise
    raise AssertionError("unreachable")


def gather(plans: Iterable[Plan[Any]]) -> Plan[list[Any]]:
    return (yield Gather(list(plans)))


def is_transient(exc: ResponseHandlingException) -> bool:
    """A transport failure without a response, e.g. a dropped connection."""
    return isinstance(exc.source, (httpx.TransportError, OSError)) and not isinstance(
        exc.source, httpx.TimeoutException
    )


def run(plan: Plan[T], execute: Callable[[Request], Any]) -> T:
    try:
        request = next(plan)
        while True:
            try:
                result = execute(request)
            except BaseException as exc:
                request = plan.throw(exc)
            else:
                request = plan.send(result)
    except StopIteration as stop:
        return cast(T, stop.value)


async def arun(plan: Plan[T], execute: Callable[[Request], Awaitable[Any]]) -> T:
    try:
        request = next(plan)
        while True:
            try:
                result = await execute(request)
            except BaseException as exc:
                request = plan.throw(exc)
            else:
                request = plan.send(result)
    except StopIteration as stop:
        return cast(T, stop.value)
