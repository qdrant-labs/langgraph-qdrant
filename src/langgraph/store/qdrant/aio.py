from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

from langchain_core.embeddings import Embeddings
from langgraph.store.base import Op, Result, TTLConfig
from langgraph.store.base.batch import AsyncBatchedBaseStore
from qdrant_client import AsyncQdrantClient

from langgraph.store.qdrant._plans import (
    Call,
    EmbedDocuments,
    EmbedQueries,
    Gather,
    Plan,
    Request,
    arun,
)
from langgraph.store.qdrant.base import (
    DEFAULT_COLLECTION_NAME,
    BaseQdrantStore,
    QdrantIndexConfig,
    _client_kwargs,
    _quiet_local_index_warning,
    _sweep_interval_seconds,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def _sweep_forever(
    ref: weakref.ReferenceType[AsyncQdrantStore], stop: asyncio.Event, interval: float
) -> None:
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        store = ref()
        if store is None:
            return
        try:
            if expired := await store.sweep_ttl():
                logger.info("Store swept %d expired items", expired)
        except Exception:
            logger.exception("Store TTL sweep iteration failed")
        del store


class AsyncQdrantStore(AsyncBatchedBaseStore, BaseQdrantStore):
    """Async long-term memory store backed by Qdrant.

    Calls made in the same event-loop tick are batched together. Must be
    created inside a running event loop. Call `setup()` once before first use.

    Examples:
        ```python
        from langgraph.store.qdrant import AsyncQdrantStore

        async with AsyncQdrantStore.from_url("http://localhost:6333") as store:
            await store.setup()
            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")
        ```
    """

    supports_ttl = True

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        index: QdrantIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        """See `QdrantStore` for the arguments."""
        super().__init__()
        self.client = client
        self._init_config(
            collection_name=collection_name,
            index=index,
            ttl=ttl,
        )
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    @classmethod
    @asynccontextmanager
    async def from_url(
        cls,
        url: str,
        *,
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        index: QdrantIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        **client_kwargs: Any,
    ) -> AsyncIterator[AsyncQdrantStore]:
        """A store with its own client, closed on exit.

        `url` is a Qdrant URL or `":memory:"`. Extra keyword arguments go to
        `AsyncQdrantClient`.
        """
        client = AsyncQdrantClient(
            location=url, api_key=api_key, **_client_kwargs(url, client_kwargs)
        )
        try:
            yield cls(
                client,
                collection_name=collection_name,
                index=index,
                ttl=ttl,
            )
        finally:
            await client.close()

    async def _run(self, plan: Plan[_T]) -> _T:
        return await arun(plan, self._execute)

    async def _execute(self, request: Request) -> Any:
        embeddings = cast(Embeddings, self.embeddings)
        match request:
            case Call(method, args, kwargs):
                return await getattr(self.client, method)(*args, **kwargs)
            case Gather(plans):
                return await asyncio.gather(*map(self._run, plans))
            case EmbedDocuments(texts):
                return await embeddings.aembed_documents(texts)
            case EmbedQueries(queries):
                return await asyncio.gather(*map(embeddings.aembed_query, queries))

    async def setup(self, **collection_options: Any) -> None:
        """Create the collection and payload indexes if they do not exist.

        Keyword arguments go to Qdrant's `create_collection` when the collection
        is created, e.g. `shard_number`, `replication_factor` and
        `write_consistency_factor`.

        Raises `ValueError` for an incompatible existing collection and
        `RuntimeError` for a server older than `MIN_SERVER_VERSION`.
        """
        with _quiet_local_index_warning():
            await self._run(self._setup_plan(collection_options))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await self._run(self._batch_plan(ops))

    async def sweep_ttl(self) -> int:
        """Delete expired items and return how many there were."""
        return await self._run(self._sweep_plan())

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: float | None = None
    ) -> asyncio.Task[None]:
        """Delete expired items periodically in a background task.

        The task holds only a weak reference to the store and stops when the
        store is garbage collected.
        """
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))
        task = self._ttl_sweeper_task
        if task is not None and not task.done():
            return task
        self._ttl_stop_event = asyncio.Event()
        interval = _sweep_interval_seconds(self.ttl_config, sweep_interval_minutes)
        task = asyncio.create_task(
            _sweep_forever(weakref.ref(self), self._ttl_stop_event, interval)
        )
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the sweeper task without cancelling a sweep in progress.

        Returns `False` if that sweep outlasts `timeout`. The task stops once
        it is done.
        """
        task = self._ttl_sweeper_task
        if task is not None and not task.done():
            self._ttl_stop_event.set()
            await asyncio.wait({task}, timeout=timeout)
            if not task.done():
                return False
        self._ttl_sweeper_task = None
        return True

    async def __aenter__(self) -> AsyncQdrantStore:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop_ttl_sweeper()

    def __del__(self) -> None:
        task = getattr(self, "_ttl_sweeper_task", None)
        if task is not None and not task.done():
            try:
                task.cancel()
            except RuntimeError:  # event loop already closed
                pass
        super().__del__()
