"""Differential fuzzing against a reference model of `BaseStore` semantics.

Random batches of puts, deletes, gets, filtered searches and namespace listings
run against the Qdrant store and a plain-Python model; every result must match.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from langgraph.store.memory import _compare_values, _does_match

from langgraph.store.qdrant import AsyncQdrantStore, QdrantStore, base
from langgraph.store.qdrant.base import point_id
from tests.conftest import inject_stale_facets, keys

LABELS = ["a", "b", "ab", "a b", "ü", "*", "x.y", '"q"', "\\", '","', "[", "null"]
KEYS = ["k1", "k2", "", " ", "ключ", '"', "a.b"]


class Model:
    """Reference implementation: reads see the state before the batch's writes."""

    def __init__(self) -> None:
        self.data: dict[tuple[tuple[str, ...], str], dict] = {}

    def batch(self, ops: list[Any]) -> list[Any]:
        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self.data.get((tuple(op.namespace), op.key)))
            elif isinstance(op, SearchOp):
                prefix = tuple(op.namespace_prefix)
                results.append(
                    {
                        (ns, key)
                        for (ns, key), value in self.data.items()
                        if ns[: len(prefix)] == prefix
                        and all(
                            _compare_values(value.get(k), v)
                            for k, v in (op.filter or {}).items()
                        )
                    }
                )
            elif isinstance(op, ListNamespacesOp):
                namespaces = {
                    ns
                    for ns, _ in self.data
                    if all(_does_match(c, ns) for c in op.match_conditions or ())
                }
                if op.max_depth is not None:
                    namespaces = {ns[: op.max_depth] for ns in namespaces}
                results.append(sorted(namespaces)[op.offset : op.offset + op.limit])
            else:
                results.append(None)
        for op in ops:
            if isinstance(op, PutOp):
                key = (tuple(op.namespace), op.key)
                if op.value is None:
                    self.data.pop(key, None)
                else:
                    self.data[key] = json.loads(json.dumps(op.value))
        return results


def _value(r: random.Random) -> dict:
    v: dict[str, Any] = {"n": r.choice([2, 3, 4, 2.5, 3.0, -7, 1e9, 0.1])}
    if r.random() < 0.9:
        v["s"] = r.choice(["red", "blue", "", "Red", "ünï", "3"])
    if r.random() < 0.7:
        v["b"] = r.random() < 0.5
    roll = r.random()
    if roll < 0.6:
        v["o"] = {"x": r.choice([1, 2, 3]), "y": r.choice(["p", "q"])}
    elif roll < 0.7:
        v["o"] = "notadict"
    v["maybe"] = r.choice([None, [], "val", "other", "<missing>"])
    if v["maybe"] == "<missing>":
        del v["maybe"]
    if r.random() < 0.5:
        v["tags"] = r.choice(
            [["x", "y"], ["y", "x"], ["x"], [], ["x", "x", "y"], [1, 1.0]]
        )
    elif r.random() < 0.5:
        v["tags"] = r.choice(["x", "y"])
    if r.random() < 0.3:
        v["big"] = r.choice([2**60, 2**60 + 1, 2**63 - 1])
    return v


def _filter(r: random.Random) -> dict:
    choices: dict[str, list[Any]] = {
        "s": ["red", {"$eq": "blue"}, {"$ne": "red"}, "", "3"],
        "n": [3, 3.0, 2.5, {"$gt": 3}, {"$gte": "3"}, {"$lt": 2.5}, {"$ne": 3},
              {"$gt": 2, "$lt": 5}, -7],
        "b": [True, False, {"$ne": True}],
        "o": [{"x": 1}, {"y": "p"}, {"x": {"$gte": 2}}, {"$eq": {"x": 1, "y": "p"}},
              {"$eq": {"x": 1}}, {"$ne": {"x": 2, "y": "q"}}, {}],
        "maybe": [None, "val", [], {"$ne": None}, {"$ne": []}],
        "tags": ["x", ["x", "y"], [], ["x"], {"$ne": ["x"]}, ["x", "x", "y"], [1, 1]],
        "big": [2**60, 2**60 + 1, {"$ne": 2**60}, 2**63 - 1],
    }  # fmt: skip
    fields = r.sample(sorted(choices), r.randint(1, 2))
    return {f: r.choice(choices[f]) for f in fields}


def _batch(r: random.Random, namespaces: list[tuple[str, ...]]) -> list[Any]:
    ops: list[Any] = []
    for _ in range(r.randint(1, 6)):
        roll = r.random()
        ns, key = r.choice(namespaces), r.choice(KEYS)
        if roll < 0.5:
            ops.append(PutOp(ns, key, _value(r)))
        elif roll < 0.6:
            ops.append(PutOp(ns, key, None))
        elif roll < 0.72:
            ops.append(GetOp(ns, key))
        elif roll < 0.88:
            ops.append(
                SearchOp(ns[: r.randint(0, len(ns))], filter=_filter(r), limit=10**5)
            )
        else:
            conditions = tuple(
                MatchCondition(
                    kind, tuple(r.choice(LABELS) for _ in range(r.randint(1, 3)))
                )
                for kind in ("prefix", "suffix")
                if r.random() < 0.5
            )
            ops.append(
                ListNamespacesOp(
                    match_conditions=conditions,
                    max_depth=r.choice([None, 1, 2]),
                    limit=r.choice([100, 2]),
                    offset=r.choice([0, 1]),
                )
            )
    return ops


def _namespaces(r: random.Random) -> list[tuple[str, ...]]:
    random_ = [
        tuple(r.choice(LABELS) for _ in range(r.randint(1, 3))) for _ in range(8)
    ]
    return [*random_, ("a",), ("a", "b"), ("ab",), ("a.b",)]


def _compare(ops: list[Any], ours: list[Any], expected: list[Any], ctx: str) -> None:
    for op, got, want in zip(ops, ours, expected, strict=True):
        if isinstance(op, GetOp):
            assert (None if got is None else got.value) == want, (ctx, op)
        elif isinstance(op, SearchOp):
            found = keys(got)
            assert len(found) == len(set(found)), (ctx, op)
            assert set(found) == want, (ctx, op)
            stamps = [i.updated_at for i in got]
            assert stamps == sorted(stamps, reverse=True), (ctx, op)
        elif isinstance(op, ListNamespacesOp):
            assert got == want, (ctx, op)


def _check_pagination(store: QdrantStore, r: random.Random) -> None:
    full = store.search((), limit=10**5)
    order = sorted(full, key=lambda i: point_id(i.namespace, i.key))
    order.sort(key=lambda i: i.updated_at, reverse=True)
    assert keys(order) == keys(full), "deterministic order"
    size = r.randint(1, 7)
    pages = [
        store.search((), limit=size, offset=o) for o in range(0, len(full) + size, size)
    ]
    assert keys(i for p in pages for i in p) == keys(full), "stable pages"


def _stale_facets(
    monkeypatch: pytest.MonkeyPatch, store: Any, namespaces: list[tuple[str, ...]]
) -> None:
    # Every namespace the run may use is reported, live or not, and small
    # chunks exercise confirming page by page.
    inject_stale_facets(monkeypatch, store.client, namespaces)
    monkeypatch.setattr(base, "_MAX_PROBES_PER_REQUEST", 3)


@pytest.mark.parametrize("seed", range(3))
def test_fuzz_sync(make_store: Any, monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    store = make_store()
    r = random.Random(seed)
    model = Model()
    namespaces = _namespaces(r)
    _stale_facets(monkeypatch, store, namespaces)
    for step in range(120):
        ops = _batch(r, namespaces)
        _compare(ops, store.batch(ops), model.batch(ops), f"seed={seed} step={step}")
        if step % 15 == 0:
            _check_pagination(store, r)
    everything = store.search((), limit=10**5)
    assert {(tuple(i.namespace), i.key): i.value for i in everything} == model.data


@pytest.mark.parametrize("seed", range(3, 5))
async def test_fuzz_async(
    make_astore: Any, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    store: AsyncQdrantStore = await make_astore()
    r = random.Random(seed)
    model = Model()
    namespaces = _namespaces(r)
    _stale_facets(monkeypatch, store, namespaces)
    for step in range(120):
        ops = _batch(r, namespaces)
        _compare(
            ops, await store.abatch(ops), model.batch(ops), f"seed={seed} step={step}"
        )
    everything = await store.asearch((), limit=10**5)
    assert {(tuple(i.namespace), i.key): i.value for i in everything} == model.data
