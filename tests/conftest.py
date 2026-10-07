from __future__ import annotations

import inspect
import math
import os
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from langgraph.store.qdrant import AsyncQdrantStore, QdrantStore, base
from langgraph.store.qdrant._payload import encode_ns
from tests.embed_test_utils import CharacterEmbeddings

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
DIMS = 64


def _server_available() -> bool:
    if os.environ.get("QDRANT_SKIP_SERVER") == "true":
        return False
    try:
        available = httpx.get(QDRANT_URL, timeout=1.0).status_code == 200
    except httpx.HTTPError:
        available = False
    if not available and os.environ.get("QDRANT_REQUIRE_SERVER") == "true":
        raise RuntimeError(f"QDRANT_REQUIRE_SERVER is set, but {QDRANT_URL} is down")
    return available


_needs_server = pytest.mark.skipif(
    not _server_available(), reason=f"no Qdrant server at {QDRANT_URL}"
)

# gRPC encodes payloads differently from REST.
CLIENTS: dict[str, dict[str, Any]] = {
    "local": {"location": ":memory:"},
    "server": {"location": QDRANT_URL},
    "grpc": {
        "location": QDRANT_URL,
        "prefer_grpc": True,
        "grpc_port": QDRANT_GRPC_PORT,
    },
}

BACKENDS = [
    "local",
    pytest.param("server", marks=_needs_server),
    pytest.param("grpc", marks=_needs_server),
]


def keys(items: Iterable[Any]) -> list[tuple[tuple[str, ...], str]]:
    return [(tuple(i.namespace), i.key) for i in items]


class SwitchableEmbeddings:
    """An embedding function that misbehaves while `fault` is set."""

    def __init__(self, dims: int = 8) -> None:
        self.fault: str | None = None
        self._embeddings = CharacterEmbeddings(dims)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embeddings.embed_documents(texts)
        match self.fault:
            case "error":
                raise RuntimeError("embedding service down")
            case "nan":
                return [[math.nan, *vector[1:]] for vector in vectors]
            case "short":
                return [vector[:-1] for vector in vectors]
        return vectors


def inject_stale_facets(
    monkeypatch: pytest.MonkeyPatch, client: Any, namespaces: Iterable[tuple[str, ...]]
) -> None:
    """Approximate facets also report `namespaces`, as if their items were
    deleted but still counted."""
    stale = [encode_ns(ns) for ns in namespaces]
    real = client.facet

    def add_stale(response: models.FacetResponse, exact: bool) -> Any:
        if exact:
            return response
        seen = {hit.value for hit in response.hits}
        extra = [models.FacetValueHit(value=v, count=1) for v in stale if v not in seen]
        return models.FacetResponse(hits=[*response.hits, *extra])

    if inspect.iscoroutinefunction(real):

        async def facet(*args: Any, **kwargs: Any) -> Any:
            return add_stale(await real(*args, **kwargs), kwargs.get("exact", False))

    else:

        def facet(*args: Any, **kwargs: Any) -> Any:
            return add_stale(real(*args, **kwargs), kwargs.get("exact", False))

    monkeypatch.setattr(client, "facet", facet)


class Clock:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = datetime.now(timezone.utc)
        monkeypatch.setattr(base, "_now", lambda: self.now)

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Freezes the store's clock. `clock.advance()` moves it."""
    return Clock(monkeypatch)


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=DIMS)


@pytest.fixture
def make_store(backend: str) -> Iterator[Any]:
    """Factory for set-up stores. Their collections are dropped afterwards."""
    client = QdrantClient(**CLIENTS[backend])
    created: list[str] = []

    def factory(**kwargs: Any) -> QdrantStore:
        kwargs.setdefault("collection_name", f"test_{uuid.uuid4().hex}")
        created.append(kwargs["collection_name"])
        store = QdrantStore(client, **kwargs)
        store.setup()
        return store

    yield factory
    for name in created:
        client.delete_collection(name)
    client.close()


@pytest.fixture
def store(make_store: Any) -> QdrantStore:
    return make_store(ttl={"refresh_on_read": True})


@pytest.fixture
async def make_astore(backend: str) -> AsyncIterator[Any]:
    client = AsyncQdrantClient(**CLIENTS[backend])
    created: list[str] = []

    async def factory(**kwargs: Any) -> AsyncQdrantStore:
        kwargs.setdefault("collection_name", f"test_{uuid.uuid4().hex}")
        created.append(kwargs["collection_name"])
        store = AsyncQdrantStore(client, **kwargs)
        await store.setup()
        return store

    yield factory
    for name in created:
        await client.delete_collection(name)
    await client.close()
