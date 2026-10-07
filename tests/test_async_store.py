"""Async-driver behaviour. The sync and fuzz tests cover store semantics."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from langgraph.store.qdrant import AsyncQdrantStore
from tests.conftest import Clock
from tests.embed_test_utils import CharacterEmbeddings


@pytest.fixture
async def astore(make_astore: Any) -> AsyncQdrantStore:
    return await make_astore(ttl={"refresh_on_read": True})


async def test_concurrent_calls_are_batched(
    astore: AsyncQdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = astore.abatch

    async def counting_abatch(ops: Any) -> Any:
        ops = list(ops)
        calls.append(len(ops))
        return await original(ops)

    monkeypatch.setattr(astore, "abatch", counting_abatch)
    await asyncio.gather(*(astore.aput(("ns",), f"k{i}", {"i": i}) for i in range(50)))
    items = await asyncio.gather(*(astore.aget(("ns",), f"k{i}") for i in range(50)))
    assert [item.value["i"] for item in items] == list(range(50))
    assert sum(calls) == 100
    assert len(calls) <= 4


async def test_async_embeddings_are_used(make_astore: Any) -> None:
    calls: list[str] = []
    sync = CharacterEmbeddings(dims=8)

    async def aembed(texts: list[str]) -> list[list[float]]:
        calls.extend(texts)
        return sync.embed_documents(texts)

    store = await make_astore(index={"dims": 8, "embed": aembed, "fields": ["t"]})
    await store.aput(("ns",), "a", {"t": "hello"})
    await store.aput(("ns",), "b", {"t": "no vector"}, index=False)
    results = await store.asearch(("ns",), query="hello")
    assert [r.key for r in results] == ["a"]
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert calls == ["hello", "hello"]


async def test_sync_call_in_event_loop_raises(astore: AsyncQdrantStore) -> None:
    with pytest.raises(asyncio.InvalidStateError):
        astore.get(("ns",), "k")


async def test_sync_calls_from_thread(astore: AsyncQdrantStore) -> None:
    def work() -> Any:
        astore.put(("ns",), "k", {"v": 1})
        return astore.get(("ns",), "k")

    item = await asyncio.to_thread(work)
    assert item.value == {"v": 1}


async def test_ttl_and_sweep(make_astore: Any, clock: Clock) -> None:
    store = await make_astore(ttl={"default_ttl": 1, "omit_expired": True})
    await store.aput(("ns",), "short", {"v": 1})
    await store.aput(("ns",), "forever", {"v": 1}, ttl=None)
    clock.advance(61)
    assert await store.aget(("ns",), "short") is None
    assert [r.key for r in await store.asearch(("ns",))] == ["forever"]
    assert await store.sweep_ttl() == 1
    assert (await store.client.count(store.collection_name)).count == 1


async def test_ttl_sweeper_task(make_astore: Any) -> None:
    store = await make_astore(ttl={"sweep_interval_minutes": 1})
    await store.aput(("ns",), "k", {}, ttl=1 / 60 / 10)
    task = await store.start_ttl_sweeper(sweep_interval_minutes=1 / 60 / 10)
    assert await store.start_ttl_sweeper() is task
    try:
        deadline = time.monotonic() + 5
        while (await store.client.count(store.collection_name)).count:
            assert time.monotonic() < deadline, "sweeper did not delete the item"
            await asyncio.sleep(0.05)
    finally:
        assert await store.stop_ttl_sweeper(timeout=5)
    assert task.done()


async def test_stop_ttl_sweeper_does_not_cancel_an_inflight_sweep(
    make_astore: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await make_astore(ttl={"sweep_interval_minutes": 1})
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_sweep() -> int:
        started.set()
        await release.wait()
        return 0

    monkeypatch.setattr(store, "sweep_ttl", slow_sweep)
    task = await store.start_ttl_sweeper(sweep_interval_minutes=1 / 60 / 100)
    await started.wait()
    assert await store.stop_ttl_sweeper(timeout=0.05) is False
    assert not task.cancelled()
    release.set()
    assert await store.stop_ttl_sweeper(timeout=5) is True
    assert task.done() and not task.cancelled()
