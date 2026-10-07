from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import httpx
import pytest
from langgraph.store.base import GetOp, PutOp
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from langgraph.store.qdrant import QdrantStore, base
from langgraph.store.qdrant._payload import MAX_VALUE_DEPTH
from langgraph.store.qdrant.base import _check_server_version, _client_kwargs, point_id
from tests.conftest import DIMS, SwitchableEmbeddings, keys
from tests.embed_test_utils import CharacterEmbeddings


def _spy(
    monkeypatch: pytest.MonkeyPatch, store: QdrantStore, method: str
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = getattr(store.client, method)

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(store.client, method, spy)
    return calls


def test_pagination_is_stable_when_timestamps_tie(store: QdrantStore) -> None:
    # One batch shares one timestamp; Qdrant does not order ties consistently.
    store.batch([PutOp(("tie",), f"t{i:03d}", {"i": i}) for i in range(60)])
    full = store.search(("tie",), limit=1000)
    assert keys(full) == sorted(keys(full), key=lambda k: point_id(*k))
    for size in (1, 7, 13):
        pages = [
            store.search(("tie",), limit=size, offset=o) for o in range(0, 60, size)
        ]
        assert keys(i for p in pages for i in p) == keys(full)


def test_pagination_is_stable_when_scores_tie(make_store: Any) -> None:
    store = make_store(
        index={"dims": DIMS, "embed": CharacterEmbeddings(DIMS), "fields": ["t"]}
    )
    ops = [PutOp(("v",), f"same{i:02d}", {"t": "tied text"}) for i in range(30)]
    ops += [
        PutOp(("v",), f"other{i}", {"t": "tied text" + "!" * i}) for i in range(1, 4)
    ]
    store.batch(ops)
    full = store.search(("v",), query="tied text", limit=100)
    assert len(full) == 33
    for size in (1, 4, 7):
        pages = [
            store.search(("v",), query="tied text", limit=size, offset=o)
            for o in range(0, 40, size)
        ]
        assert keys(i for p in pages for i in p) == keys(full)


def test_large_tie_groups_read_only_what_the_page_needs(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.batch([PutOp(("t",), f"k{i:03d}", {"i": i}) for i in range(200)])
    calls = _spy(monkeypatch, store, "scroll")
    page = store.search(("t",), limit=10, offset=20)
    assert sum(c["limit"] for c in calls) <= 30
    assert keys(page) == keys(store.search(("t",), limit=200)[20:30])


def test_deep_pages_fetch_only_ranking_fields(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.batch(
        [PutOp(("p",), f"k{i:03d}", {"i": i, "blob": "x" * 1000}) for i in range(50)]
    )
    calls = _spy(monkeypatch, store, "query_batch_points")
    page = store.search(("p",), limit=5, offset=40)
    assert len(page) == 5 and all(i.value["blob"] for i in page)
    (request,) = calls[0]["requests"]
    assert request.with_payload == ["updated_at"]


def test_deleted_namespaces_are_not_listed(store: QdrantStore) -> None:
    store.batch([PutOp(("ns", str(i)), "k", {}) for i in range(20)])
    store.batch([PutOp(("ns", str(i)), "k", None) for i in range(0, 20, 2)])
    expected = sorted(("ns", str(i)) for i in range(1, 20, 2))
    assert store.list_namespaces(limit=100) == expected
    assert store.list_namespaces(prefix=("ns",), limit=100) == expected


@pytest.mark.parametrize(
    "version,supported",
    [
        ("1.16.0", True),
        ("1.19.2", True),
        ("2.0.0-dev", True),
        ("unknown", True),
        ("1.15.5", False),
    ],
)
def test_server_version_check(version: str, supported: bool) -> None:
    if supported:
        _check_server_version(version)
    else:
        with pytest.raises(RuntimeError, match="requires 1.16.0"):
            _check_server_version(version)


def test_namespace_labels_with_dots_do_not_collide(store: QdrantStore) -> None:
    # `batch()` bypasses `put()` validation, so labels may contain dots.
    store.batch(
        [
            PutOp(("a.b",), "k", {"v": 1}),
            PutOp(("a", "b"), "k", {"v": 2}),
            PutOp(('a","b',), "k", {"v": 3}),
        ]
    )
    assert [i.value for i in store.search(("a.b",))] == [{"v": 1}]
    assert [i.value for i in store.search(("a", "b"))] == [{"v": 2}]
    assert [i.value for i in store.search(('a","b',))] == [{"v": 3}]
    assert store.list_namespaces(prefix=("a",)) == [("a", "b")]
    assert sorted(store.list_namespaces()) == sorted([("a.b",), ("a", "b"), ('a","b',)])


@pytest.mark.parametrize(
    "value,exc",
    [
        ({"d": datetime(2026, 1, 1)}, TypeError),
        ({"x": float("nan")}, ValueError),
        ({"x": 2**64}, ValueError),
        ({"x": 2**63}, ValueError),
        ({"x": -(2**63) - 1}, ValueError),
        ({"x": {1, 2}}, TypeError),
    ],
)
def test_unstorable_values_fail_without_side_effects(
    store: QdrantStore, value: dict, exc: type[Exception]
) -> None:
    store.put(("v",), "keep", {"x": 1})
    with pytest.raises(exc):
        store.batch(
            [
                PutOp(("v",), "keep", None),
                PutOp(("v",), "new", {"ok": True}),
                PutOp(("v",), "bad", value),
            ]
        )
    assert store.get(("v",), "keep").value == {"x": 1}
    assert store.get(("v",), "new") is None


def test_value_depth_limit(store: QdrantStore) -> None:
    ok: Any = "leaf"
    for _ in range(MAX_VALUE_DEPTH):  # the top-level dict is level 1
        ok = {"d": ok}
    store.put(("v",), "ok", ok)
    assert store.get(("v",), "ok").value == ok
    with pytest.raises(ValueError, match="nested deeper"):
        store.put(("v",), "deep", {"d": ok})


def test_values_are_json_normalized(store: QdrantStore) -> None:
    store.put(("v",), "k", {"t": (1, 2), 3: "x", "big": 2**63 - 1, "neg": -(2**63)})
    assert store.get(("v",), "k").value == {
        "t": [1, 2],
        "3": "x",
        "big": 2**63 - 1,
        "neg": -(2**63),
    }


def test_values_round_trip_exactly(store: QdrantStore) -> None:
    # A Qdrant server changes some floats by one ULP, and gRPC reorders keys.
    value = {"z": 123456789.12345679, "a": [9007199254740991.0, 0.1 + 0.2], "m": 1}
    store.put(("v",), "k", value)
    for item in [store.get(("v",), "k"), *store.search(("v",))]:
        assert json.dumps(item.value) == json.dumps(value)


FILTER_DOCS: dict[str, dict[Any, Any]] = {
    "empty_list": {"f": []},
    "null": {"f": None},
    "missing": {},
    "empty_obj": {"f": {}},
    "str": {"f": "x"},
    "list": {"f": ["x", "y"]},
    "obj": {"f": {"a": 1}},
    "big": {"f": 2**60 + 1},
    "max": {"f": 2**63 - 1},
    "edge": {"f": 2**53 - 1},
    "float_big": {"f": float(2**60)},
    "imprecise": {"f": 123456789.12345679},  # Qdrant parses it one ULP off
    "num": {"f": 20240602},
    "date": {"f": "2024-06-02T00:00:00+00:00"},
    "intkey": {1: "one"},
    "newline": {"a\n": 1, "a": 2},
}

EXACT_FILTER_CASES: list[tuple[dict[Any, Any], set[str]]] = [
    ({"f": None}, {"null", "missing", "intkey", "newline"}),
    ({"f": []}, {"empty_list"}),
    ({"f": "x"}, {"str"}),
    ({"f": ["x", "y"]}, {"list"}),
    ({"f": ["y", "x"]}, set()),
    ({"f": {}}, {"empty_obj", "obj"}),
    ({"f": {"a": None}}, {"empty_obj"}),
    ({"f": {"$eq": {"a": 1.0}}}, {"obj"}),
    ({"f": 2**60 + 1}, {"big"}),
    ({"f": 2**60}, {"float_big"}),
    ({"f": float(2**60)}, {"float_big"}),
    ({"f": 2**63}, set()),
    ({"f": 2**53 - 1}, {"edge"}),
    ({"f": 123456789.12345679}, {"imprecise"}),
    ({"f": {"$gte": 900719925474099}}, {"big", "max", "edge", "float_big"}),
    ({"f": {"$lt": -1e300}}, set()),
    ({"f": {"$lt": 1e300}}, {"num", "big", "max", "edge", "float_big", "imprecise"}),
    ({"f": 2**63 - 1}, {"max"}),
    ({"f": {"$ne": 2**63 - 1}}, set(FILTER_DOCS) - {"max"}),
    *(
        ({"f": {"$gt": bound}}, {"num", "big", "max", "edge", "float_big", "imprecise"})
        for bound in ["20240601", " 1e3 "]
    ),
    ({"f": {"$gt": "2024-06-01"}}, {"date"}),
    ({"f": {"$gt": "2024-06-01T00:00:00Z"}}, {"date"}),
    ({1: "one"}, {"intkey"}),
    ({"a\n": 1}, {"newline"}),
    ({"a\n": 2}, set()),
]

FILTER_ERRORS: list[tuple[dict[str, Any], str]] = [
    ({"f": {"$gte": 2**60 + 1}}, "exactly"),
    ({"f": {"$lt": 2**53 - 1}}, "exactly"),  # sent as "...991.0", read as ...990
    ({"f": {"$lt": 1e19}}, "exactly"),  # floats are accepted beyond ±2**64
    ({"back\\slash": 1}, "backslashes"),
    *(
        ({"f": {"$gt": operand}}, "finite number or an ISO-8601")
        for operand in ["1_000", "nan", "inf", "Infinity", "abc", ""]
    ),
]


def test_exact_filter_semantics(store: QdrantStore) -> None:
    store.batch([PutOp(("d",), key, value) for key, value in FILTER_DOCS.items()])
    for flt, expected in EXACT_FILTER_CASES:
        assert {
            i.key for i in store.search(("d",), filter=flt, limit=100)
        } == expected, flt
    for flt, message in FILTER_ERRORS:
        with pytest.raises(ValueError, match=message):
            store.search(("d",), filter=flt)


def test_requests_stay_under_the_size_budget(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `value_paths` is several times larger than this value, so an estimate
    # that ignored it would overflow the budget.
    budget = 200_000
    monkeypatch.setattr(base, "_MAX_REQUEST_BYTES", budget)
    calls = _spy(monkeypatch, store, "upsert")
    value = {"users": {f"user{i}": {"name": "a"} for i in range(300)}}
    store.batch([PutOp(("big",), f"k{i}", value) for i in range(30)])
    assert len(calls) > 1
    for call in calls:
        points = [p.model_dump(mode="json") for p in call["points"]]
        assert len(json.dumps(points)) <= budget
    assert len(store.search(("big",), limit=100)) == 30


def test_large_batches_are_chunked(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base, "_MAX_POINTS_PER_REQUEST", 7)
    monkeypatch.setattr(base, "_MAX_IDS_PER_REQUEST", 5)
    upserts = _spy(monkeypatch, store, "upsert")
    retrieves = _spy(monkeypatch, store, "retrieve")
    deletes = _spy(monkeypatch, store, "delete")
    store.batch([PutOp(("many",), f"k{i}", {"i": i}) for i in range(40)])
    got = store.batch([GetOp(("many",), f"k{i}") for i in range(40)])
    assert [g.value["i"] for g in got] == list(range(40))
    store.batch([PutOp(("many",), f"k{i}", None) for i in range(40)])
    assert store.search(("many",)) == []
    assert [len(c["points"]) for c in upserts] == [7] * 5 + [5]
    assert max(len(c["ids"]) for c in retrieves) == 5
    assert [len(c["points_selector"].points) for c in deletes] == [5] * 8


def test_oversized_item_is_rejected_before_writing(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base, "MAX_ITEM_BYTES", 10_000)
    store.put(("v",), "keep", {"x": 1})
    with pytest.raises(ValueError, match="single Qdrant request"):
        store.batch(
            [PutOp(("v",), "keep", None), PutOp(("v",), "huge", {"b": "x" * 20_000})]
        )
    assert store.get(("v",), "keep") is not None


EMBEDDING_FAULTS = [
    ("error", RuntimeError, "service down"),
    ("nan", ValueError, "NaN"),
    ("short", ValueError, "configured for 8 dims"),
]


@pytest.mark.parametrize("fault,error,message", EMBEDDING_FAULTS)
def test_embedding_failure_writes_nothing(
    make_store: Any, fault: str, error: type[Exception], message: str
) -> None:
    embed = SwitchableEmbeddings()
    store = make_store(index={"dims": 8, "embed": embed, "fields": ["t"]})
    store.put(("e",), "keep", {"t": "a"})
    embed.fault = fault
    with pytest.raises(error, match=message):
        store.batch([PutOp(("e",), "keep", None), PutOp(("e",), "new", {"t": "b"})])
    with pytest.raises(error, match=message):
        store.search(("e",), query="a")
    embed.fault = None
    assert store.get(("e",), "keep") is not None
    assert store.get(("e",), "new") is None


@pytest.mark.parametrize("fault,error,message", EMBEDDING_FAULTS)
def test_failed_embedding_does_not_refresh_ttl(
    make_store: Any, fault: str, error: type[Exception], message: str
) -> None:
    embed = SwitchableEmbeddings()
    store = make_store(
        index={"dims": 8, "embed": embed, "fields": ["t"]},
        ttl={"refresh_on_read": True},
    )
    store.put(("t",), "a", {"t": "x"}, ttl=10)

    def expires_at() -> str:
        record = store.client.retrieve(store.collection_name, [point_id(("t",), "a")])[
            0
        ]
        return record.payload["expires_at"]

    before = expires_at()
    embed.fault = fault
    with pytest.raises(error, match=message):
        store.batch([GetOp(("t",), "a"), PutOp(("t",), "b", {"t": "y"})])
    assert expires_at() == before


@pytest.mark.parametrize(
    "error,failures,expected_calls",
    [
        (ResponseHandlingException(httpx.ReadError("Bad file descriptor")), 2, 3),
        (ResponseHandlingException(httpx.ConnectError("refused")), math.inf, 3),
        (ResponseHandlingException(httpx.ReadTimeout("slow")), math.inf, 1),
        (UnexpectedResponse(500, "err", b"", httpx.Headers()), math.inf, 1),
    ],
    ids=["transient-recovers", "transient-gives-up", "timeout", "http-error"],
)
def test_transport_retries(
    store: QdrantStore,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    failures: float,
    expected_calls: int,
) -> None:
    store.put(("r",), "k", {"v": 1})
    real = store.client.retrieve
    calls = 0

    def flaky(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise error
        return real(*args, **kwargs)

    monkeypatch.setattr(store.client, "retrieve", flaky)
    if failures < expected_calls:
        assert store.get(("r",), "k").value == {"v": 1}
    else:
        with pytest.raises(type(error)):
            store.get(("r",), "k")
    assert calls == expected_calls


@pytest.mark.parametrize(
    "url,pooled",
    [
        ("http://localhost:6333", True),
        ("localhost:6333", True),
        ("localhost", True),
        ("127.0.0.1:6333", True),
        ("http://[::1]:6333", True),
        ("https://xyz.cloud.qdrant.io", False),
        ("qdrant.internal:6333", False),
        (":memory:", False),
    ],
)
def test_from_url_pools_local_connections_only(url: str, pooled: bool) -> None:
    assert ("limits" in _client_kwargs(url, {})) is pooled
    assert _client_kwargs(url, {"pool_size": 4}) == {"pool_size": 4}
