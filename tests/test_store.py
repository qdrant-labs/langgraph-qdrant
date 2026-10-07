from __future__ import annotations

import gc
import math
import time
from datetime import datetime, timezone
from typing import Any

import pytest
from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from langgraph.store.memory import InMemoryStore
from qdrant_client import models

from langgraph.store.qdrant import QdrantStore, base
from langgraph.store.qdrant.base import VECTOR_NAME, point_id
from tests.conftest import DIMS, Clock, inject_stale_facets
from tests.embed_test_utils import CharacterEmbeddings


def test_point_id_is_deterministic() -> None:
    assert point_id(("a", "b"), "k") == point_id(["a", "b"], "k")
    assert point_id(("a", "b"), "k") != point_id(("a",), "b.k")
    assert point_id(("a", "b"), "k") != point_id(("a.b",), "k")


def test_basic_store_ops(store: QdrantStore) -> None:
    namespace = ("test", "documents")
    store.put(namespace, "doc1", {"title": "Doc 1", "content": "Hello"})
    item = store.get(namespace, "doc1")
    assert isinstance(item, Item)
    assert item.namespace == namespace
    assert item.key == "doc1"
    assert item.value == {"title": "Doc 1", "content": "Hello"}
    assert item.created_at.tzinfo is not None
    assert item.created_at == item.updated_at

    time.sleep(0.01)
    store.put(namespace, "doc1", {"title": "Doc 1 v2"})
    updated = store.get(namespace, "doc1")
    assert updated is not None
    assert updated.value == {"title": "Doc 1 v2"}
    assert updated.created_at == item.created_at
    assert updated.updated_at > item.updated_at

    assert store.get(namespace, "missing") is None
    assert store.get(("other",), "doc1") is None

    store.delete(namespace, "doc1")
    assert store.get(namespace, "doc1") is None
    store.delete(namespace, "never-existed")


def test_rejects_invalid_namespaces(store: QdrantStore) -> None:
    for namespace in [(), ("a.b",), ("langgraph", "x")]:
        with pytest.raises(InvalidNamespaceError):
            store.put(namespace, "k", {})


def test_batch_order(store: QdrantStore) -> None:
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})

    results = store.batch(
        [
            GetOp(namespace=("test", "foo"), key="key1"),
            PutOp(namespace=("test", "bar"), key="key2", value={"data": "new"}),
            SearchOp(namespace_prefix=("test",), filter={"data": "value1"}),
            ListNamespacesOp(match_conditions=None, max_depth=None),
            GetOp(namespace=("test",), key="key3"),
            GetOp(namespace=("test", "foo"), key="key1"),
        ]
    )
    assert isinstance(results[0], Item) and results[0].value == {"data": "value1"}
    assert results[1] is None
    assert [r.key for r in results[2]] == ["key1"]
    assert results[3] == [("test", "bar"), ("test", "foo")]
    assert results[4] is None
    assert results[5] == results[0]
    # Writes are applied after the reads of the same batch.
    assert store.get(("test", "bar"), "key2").value == {"data": "new"}


def test_batch_dedupes_puts_last_wins(store: QdrantStore) -> None:
    store.batch(
        [
            PutOp(("ns",), "k", {"v": 1}),
            PutOp(("ns",), "k", {"v": 2}),
            PutOp(("ns",), "gone", {"v": 1}),
            PutOp(("ns",), "gone", None),
        ]
    )
    assert store.get(("ns",), "k").value == {"v": 2}
    assert store.get(("ns",), "gone") is None


def test_non_ascii_and_special_labels(store: QdrantStore) -> None:
    namespaces = [
        ("user", "日本語"),
        ("user", "with space"),
        ("user", "100%_like"),
        ("user", "trailing\n"),
    ]
    for i, ns in enumerate(namespaces):
        store.put(ns, f"key-{i}", {"name": "José", "emoji": "🚀"})
    for i, ns in enumerate(namespaces):
        assert store.get(ns, f"key-{i}").value["emoji"] == "🚀"
    assert sorted(store.list_namespaces(prefix=("user",))) == sorted(namespaces)
    assert [r.key for r in store.search(("user", "100%_like"))] == ["key-2"]


NAMESPACES = [
    ("a", "b", "c"),
    ("a", "b", "d", "e"),
    ("a", "b", "d", "i"),
    ("a", "b", "f"),
    ("a", "c", "f"),
    ("b", "a", "f"),
    ("users", "123"),
    ("users", "456", "settings"),
    ("admin", "users", "789"),
    ("ab", "x"),
]

LIST_CASES: list[dict[str, Any]] = [
    {},
    {"prefix": ("a",)},
    {"prefix": ("a", "b")},
    {"prefix": ("a", "b"), "max_depth": 3},
    {"prefix": ("a", "*")},
    {"prefix": ("a", "*", "f")},
    {"prefix": ("*", "*", "f")},
    {"suffix": ("f",)},
    {"suffix": ("d", "*")},
    {"suffix": ("*", "f")},
    {"prefix": ("a",), "suffix": ("f",)},
    {"prefix": ("users",), "max_depth": 2},
    {"max_depth": 1},
    {"max_depth": 2, "limit": 3},
    {"limit": 4, "offset": 3},
    {"prefix": ("nonexistent",)},
    {"prefix": ("a", "b", "c", "d", "e", "f")},
]


def test_list_namespaces_matches_in_memory(store: QdrantStore) -> None:
    reference = InMemoryStore()
    store.batch([PutOp(ns, f"k{i}", {"i": i}) for i, ns in enumerate(NAMESPACES)])
    for i, ns in enumerate(NAMESPACES):
        reference.put(ns, f"k{i}", {"i": i})
    for case in LIST_CASES:
        expected = reference.list_namespaces(**case)
        assert store.list_namespaces(**case) == expected, case


def test_list_namespaces_segment_boundary(store: QdrantStore) -> None:
    store.put(("a", "b"), "k", {})
    store.put(("a", "bc"), "k", {})
    store.put(("ab",), "k", {})
    assert store.list_namespaces(prefix=("a", "b")) == [("a", "b")]
    assert store.list_namespaces(prefix=("a",)) == [("a", "b"), ("a", "bc")]
    assert store.list_namespaces(suffix=("b",)) == [("a", "b")]


def test_list_namespaces_ops(store: QdrantStore) -> None:
    store.batch([PutOp(ns, "k", {}) for ns in NAMESPACES])
    results = store.batch(
        [
            ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("users",)),)),
            ListNamespacesOp(
                match_conditions=(MatchCondition("suffix", ("f",)),), max_depth=2
            ),
        ]
    )
    assert results[0] == [("users", "123"), ("users", "456", "settings")]
    assert results[1] == [("a", "b"), ("a", "c"), ("b", "a")]


def test_list_namespaces_skips_stale_facet_hits(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base, "_MAX_PROBES_PER_REQUEST", 2)
    live = [("ns", str(i)) for i in range(5)] + [("ns", "x", "deep")]
    store.batch([PutOp(ns, "k", {}) for ns in [*live, ("other",)]])
    stale = [("ns", f"{i}gone") for i in range(5)] + [("ns", "x", "stale")]
    inject_stale_facets(monkeypatch, store.client, stale)

    expected = sorted(live)
    assert store.list_namespaces(prefix=("ns",)) == expected
    assert store.list_namespaces(prefix=("ns",), limit=3, offset=2) == expected[2:5]
    assert store.list_namespaces(prefix=("ns",), max_depth=2) == sorted(
        [*live[:5], ("ns", "x")]
    )
    assert len(store.list_namespaces()) == len(live) + 1


DOCS: list[tuple[str, dict[str, Any]]] = [
    ("d1", {"color": "red", "score": 4.5, "count": 3, "active": True,
            "meta": {"lang": "en", "level": 1}, "tags": ["x", "y"],
            "when": "2024-01-01T00:00:00+00:00"}),
    ("d2", {"color": "blue", "score": 3.0, "count": 10, "active": False,
            "meta": {"lang": "fr", "level": 2}, "tags": ["y"],
            "when": "2024-06-01T00:00:00+00:00"}),
    ("d3", {"color": "red", "score": 5.0, "count": 1, "active": True,
            "meta": {"lang": "en", "level": 3}, "tags": ["y", "x"],
            "when": "2025-01-01T00:00:00+00:00", "note": None}),
    ("d4", {"color": "green", "score": 1, "count": 1.0, "active": False,
            "meta": {"lang": "de", "level": 1}, "tags": []}),
]  # fmt: skip

FILTER_CASES: list[tuple[dict[str, Any], set[str]]] = [
    ({"color": "red"}, {"d1", "d3"}),
    ({"color": {"$eq": "blue"}}, {"d2"}),
    ({"color": {"$ne": "red"}}, {"d2", "d4"}),
    ({"score": {"$gt": 4.0}}, {"d1", "d3"}),
    ({"score": {"$gte": 4.5}}, {"d1", "d3"}),
    ({"score": {"$lt": 3.0}}, {"d4"}),
    ({"score": {"$lte": 3.0}}, {"d2", "d4"}),
    ({"score": {"$gt": 2, "$lt": 5}}, {"d1", "d2"}),
    ({"count": 1}, {"d3", "d4"}),  # int matches float 1.0
    ({"score": 3}, {"d2"}),
    ({"active": True}, {"d1", "d3"}),
    ({"active": False, "color": "green"}, {"d4"}),
    ({"meta": {"lang": "en"}}, {"d1", "d3"}),
    ({"meta": {"lang": "en", "level": 3}}, {"d3"}),
    ({"meta": {"level": {"$gte": 2}}}, {"d2", "d3"}),
    ({"tags": ["y"]}, {"d2"}),
    ({"tags": ["x", "y"]}, {"d1"}),  # order matters
    ({"tags": []}, {"d4"}),
    ({"tags": "y"}, set()),  # a scalar never matches an array
    ({"tags": {"$ne": ["y"]}}, {"d1", "d3", "d4"}),
    ({"note": None}, {"d1", "d2", "d3", "d4"}),  # missing or null
    ({"meta": {"$eq": {"lang": "en", "level": 1}}}, {"d1"}),
    ({"meta": {"$eq": {"lang": "en"}}}, set()),
    ({"meta": {"$ne": {"lang": "fr", "level": 2.0}}}, {"d1", "d3", "d4"}),
    ({"meta": {}}, {"d1", "d2", "d3", "d4"}),
    ({"when": {"$gte": "2024-06-01T00:00:00+00:00"}}, {"d2", "d3"}),
    ({"when": {"$lt": datetime(2024, 3, 1, tzinfo=timezone.utc)}}, {"d1"}),
    ({"color": "purple"}, set()),
]


def test_search_filters(store: QdrantStore) -> None:
    store.batch([PutOp(("docs",), key, value) for key, value in DOCS])
    store.put(("other",), "d1", DOCS[0][1])
    for flt, expected in FILTER_CASES:
        results = store.search(("docs",), filter=flt, limit=10)
        assert {r.key for r in results} == expected, flt
        assert all(r.score is None for r in results)


def test_search_filter_errors(store: QdrantStore) -> None:
    errors = [
        ({"a": {"$in": [1]}}, "Unsupported operator"),
        ({"a": {"$gt": "abc"}}, "number or an ISO-8601"),
        ({"a": [float("nan")]}, "valid JSON"),
        ({"a": float("inf")}, "finite"),
    ]
    for flt, match in errors:
        with pytest.raises(ValueError, match=match):
            store.search(("docs",), filter=flt)


def test_search_keys_needing_quotes(store: QdrantStore) -> None:
    store.put(("docs",), "a", {"dotted.key": 1, "with space": "x"})
    store.put(("docs",), "b", {"dotted": {"key": 1}})
    assert [r.key for r in store.search(("docs",), filter={"dotted.key": 1})] == ["a"]
    assert [r.key for r in store.search(("docs",), filter={"with space": "x"})] == ["a"]


def test_search_orders_by_updated_at_and_paginates(store: QdrantStore) -> None:
    for i in range(5):
        store.put(("docs",), f"d{i}", {"i": i})
        time.sleep(0.005)
    store.put(("docs",), "d0", {"i": 0, "touched": True})
    assert [r.key for r in store.search(("docs",))] == ["d0", "d4", "d3", "d2", "d1"]
    assert [r.key for r in store.search(("docs",), limit=2, offset=1)] == ["d4", "d3"]
    assert store.search(("docs",), limit=2, offset=10) == []


@pytest.mark.parametrize(
    "op",
    [
        SearchOp(("ns",), offset=-1),
        SearchOp(("ns",), limit=-1),
        ListNamespacesOp(offset=-1),
        ListNamespacesOp(limit=-1),
    ],
)
def test_negative_offset_or_limit_is_rejected(store: QdrantStore, op: Any) -> None:
    store.put(("ns",), "k", {})
    with pytest.raises(ValueError, match="non-negative"):
        store.batch([PutOp(("ns",), "new", {}), op])
    assert store.get(("ns",), "new") is None


def test_search_with_zero_limit(store: QdrantStore) -> None:
    store.put(("ns",), "k", {})
    results = store.batch([SearchOp(("ns",), limit=0), SearchOp(("ns",), limit=1)])
    assert results[0] == []
    assert [r.key for r in results[1]] == ["k"]


def test_search_namespace_prefix(store: QdrantStore) -> None:
    store.put(("a", "b"), "1", {})
    store.put(("a", "bc"), "2", {})
    store.put(("a", "b", "c"), "3", {})
    store.put(("x",), "4", {})
    assert {r.key for r in store.search(("a", "b"))} == {"1", "3"}
    assert {r.key for r in store.search(("a",))} == {"1", "2", "3"}
    assert {r.key for r in store.search(())} == {"1", "2", "3", "4"}


def test_query_without_index_falls_back_to_filtering(store: QdrantStore) -> None:
    store.put(("docs",), "a", {"text": "hello"})
    results = store.search(("docs",), query="hello")
    assert [(r.key, r.score) for r in results] == [("a", None)]


def _similarity(kind: str, a: list[float], b: list[float]) -> float:
    pairs = list(zip(a, b, strict=True))
    if kind == "cosine":
        norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        return sum(x * y for x, y in pairs) / norms
    if kind == "dot":
        return sum(x * y for x, y in pairs)
    if kind == "euclid":
        return -math.sqrt(sum((x - y) ** 2 for x, y in pairs))
    return -sum(abs(x - y) for x, y in pairs)


@pytest.fixture(
    params=[{}, {"hnsw_config": models.HnswConfigDiff(m=0, payload_m=16)}],
    ids=["global-hnsw", "per-tenant-hnsw"],
)
def vector_store(
    make_store: Any, fake_embeddings: CharacterEmbeddings, request: Any
) -> QdrantStore:
    return make_store(
        index={
            "dims": DIMS,
            "embed": fake_embeddings,
            "fields": ["text"],
            **request.param,
        }
    )


def test_vector_store_initialization(
    vector_store: QdrantStore, fake_embeddings: CharacterEmbeddings
) -> None:
    assert vector_store.embeddings is fake_embeddings
    info = vector_store.client.get_collection(vector_store.collection_name)
    params = info.config.params.vectors[VECTOR_NAME]
    assert params.size == DIMS
    assert params.multivector_config is not None
    vector_store.setup()  # idempotent


def test_vector_insert_and_search(vector_store: QdrantStore) -> None:
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
        ("doc4", {"description": "text in description field"}),
        ("doc5", {"text": "unrelated content"}),
    ]
    vector_store.batch([PutOp(("test",), key, value) for key, value in docs])

    results = vector_store.search(("test",), query="long text")
    keys = [r.key for r in results]
    assert "doc4" not in keys  # no `text` field, so no vector
    assert keys[0] in {"doc2", "doc3"}
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

    records = vector_store.client.retrieve(
        vector_store.collection_name, [point_id(("test",), "doc4")], with_vectors=True
    )
    assert not records[0].vector


def test_vector_update_changes_embedding(vector_store: QdrantStore) -> None:
    vector_store.put(("test",), "doc1", {"text": "zany zebra zoo"})
    before = vector_store.search(("test",), query="zany zebra zoo")[0].score
    vector_store.put(("test",), "doc1", {"text": "quiet quokka"})
    after = vector_store.search(("test",), query="zany zebra zoo")[0].score
    assert after < before
    assert vector_store.get(("test",), "doc1").value == {"text": "quiet quokka"}


def test_vector_index_false_removes_vector(vector_store: QdrantStore) -> None:
    vector_store.put(("test",), "doc1", {"text": "hello"})
    assert vector_store.search(("test",), query="hello")
    vector_store.put(("test",), "doc1", {"text": "hello"}, index=False)
    assert vector_store.search(("test",), query="hello") == []
    assert vector_store.search(("test",))[0].key == "doc1"


def test_vector_search_with_filters_and_pagination(vector_store: QdrantStore) -> None:
    docs = [
        ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
        ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
        ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
        ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
    ]
    vector_store.batch([PutOp(("test",), key, value) for key, value in docs])

    results = vector_store.search(("test",), query="apple", filter={"color": "red"})
    assert [r.key for r in results] == ["doc1", "doc2"]

    results = vector_store.search(
        ("test",), query="car", filter={"score": {"$gte": 3.5}}
    )
    assert results[0].key == "doc4"
    assert {r.key for r in results} == {"doc1", "doc3", "doc4"}

    all_results = vector_store.search(("test",), query="apple", limit=4)
    page = vector_store.search(("test",), query="apple", limit=2, offset=1)
    assert [r.key for r in page] == [r.key for r in all_results[1:3]]


@pytest.mark.parametrize("distance", ["cosine", "dot", "euclid", "manhattan"])
def test_scores_match_bruteforce_max_pooling(
    make_store: Any, fake_embeddings: CharacterEmbeddings, distance: str
) -> None:
    store = make_store(
        index={
            "dims": DIMS,
            "embed": fake_embeddings,
            "fields": ["key0", "key1", "items[*].text"],
            "distance": distance,
        }
    )
    value = {
        "key0": "aaa",
        "key1": "bbb ccc",
        "items": [{"text": "abcd"}, {"text": "poison"}],
    }
    store.put(("test",), "doc", value)
    store.put(("test",), "other", {"key0": "zzz"})

    for query in ["aaa", "bbb", "abcd", "poisson"]:
        results = store.search(("test",), query=query)
        q = fake_embeddings.embed_query(query)
        expected = max(
            _similarity(distance, q, fake_embeddings.embed_query(text))
            for text in ["aaa", "bbb ccc", "abcd", "poison"]
        )
        doc = next(r for r in results if r.key == "doc")
        assert doc.score == pytest.approx(expected, rel=1e-4, abs=1e-4), query
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


def test_embed_with_path_and_index_override(
    make_store: Any, fake_embeddings: CharacterEmbeddings
) -> None:
    store = make_store(
        index={"dims": DIMS, "embed": fake_embeddings, "fields": ["title"]}
    )
    store.put(("docs",), "a", {"title": "xxxx", "body": "qqqq"})
    store.put(("docs",), "b", {"title": "zzzz", "body": "xxxx"}, index=["body"])
    store.put(
        ("docs",),
        "c",
        {"chapters": [{"t": "wwww"}, {"t": "xxxx"}]},
        index=["chapters[*].t"],
    )

    by_key = {r.key: r.score for r in store.search(("docs",), query="xxxx")}
    assert by_key == pytest.approx({"a": 1.0, "b": 1.0, "c": 1.0}, abs=1e-4)
    # "a" was indexed on its title only, so its body text is not searchable.
    scores = {r.key: r.score for r in store.search(("docs",), query="qqqq")}
    assert scores["a"] < 0.99
    records = store.client.retrieve(
        store.collection_name, [point_id(("docs",), "c")], with_vectors=True
    )
    assert len(records[0].vector[VECTOR_NAME]) == 2


def test_whole_document_embedding_by_default(
    make_store: Any, fake_embeddings: CharacterEmbeddings
) -> None:
    store = make_store(index={"dims": DIMS, "embed": fake_embeddings})
    store.put(("docs",), "a", {"text": "hello"})
    records = store.client.retrieve(
        store.collection_name, [point_id(("docs",), "a")], with_vectors=True
    )
    assert len(records[0].vector[VECTOR_NAME]) == 1
    score = store.search(("docs",), query='{"text": "hello"}')[0].score
    assert score == pytest.approx(1.0, abs=1e-4)


def test_embeddings_called_once_per_unique_text(make_store: Any) -> None:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return CharacterEmbeddings(dims=8).embed_documents(texts)

    store = make_store(index={"dims": 8, "embed": embed, "fields": ["t"]})
    store.batch(
        [
            PutOp(("ns",), "a", {"t": "same"}),
            PutOp(("ns",), "b", {"t": "same"}),
            PutOp(("ns",), "c", {"t": "other"}),
        ]
    )
    assert calls == [["same", "other"]]


def test_setup_rejects_incompatible_collection(
    make_store: Any, fake_embeddings: CharacterEmbeddings
) -> None:
    plain = make_store()
    with pytest.raises(ValueError, match="no 'embedding' vector"):
        QdrantStore(
            plain.client,
            collection_name=plain.collection_name,
            index={"dims": DIMS, "embed": fake_embeddings},
        ).setup()

    vectors = make_store(index={"dims": DIMS, "embed": fake_embeddings})
    with pytest.raises(ValueError, match="configured for size 32"):
        QdrantStore(
            vectors.client,
            collection_name=vectors.collection_name,
            index={"dims": 32, "embed": CharacterEmbeddings(dims=32)},
        ).setup()


def test_setup_creates_tenant_indexes(store: QdrantStore, backend: str) -> None:
    if backend == "local":
        pytest.skip("local mode has no payload indexes")
    schema = store.client.get_collection(store.collection_name).payload_schema
    params = {field: info.params for field, info in schema.items()}
    assert params["ns_path"].is_tenant
    if any(p.enable_hnsw is not None for p in params.values()):  # Qdrant 1.17+
        with_graphs = {f for f, p in params.items() if p.enable_hnsw is not False}
        assert with_graphs == {"ns_prefixes"}


def test_setup_passes_collection_options(make_store: Any, backend: str) -> None:
    store = make_store(index={"dims": DIMS, "embed": CharacterEmbeddings(DIMS)})
    name = f"{store.collection_name}_sharded"
    sharded = QdrantStore(store.client, collection_name=name, index=store.index_config)
    sharded.setup(shard_number=2, on_disk_payload=True)
    try:
        sharded.put(("ns",), "k", {"text": "hello"})
        assert sharded.search(("ns",), query="hello")[0].key == "k"
        if backend != "local":  # local mode has no shards
            params = store.client.get_collection(name).config.params
            assert (params.shard_number, params.on_disk_payload) == (2, True)
    finally:
        store.client.delete_collection(name)


def test_setup_checks_the_data_format(make_store: Any) -> None:
    store = make_store()
    client, name = store.client, store.collection_name
    assert client.get_collection(name).config.metadata == {
        "langgraph_store_format": base.FORMAT_VERSION
    }

    # A collection created by hand is adopted while it is empty.
    client.update_collection(name, metadata={"langgraph_store_format": None})
    QdrantStore(client, collection_name=name).setup()
    store.put(("ns",), "k", {"v": 1})
    QdrantStore(client, collection_name=name).setup()

    for stamp, message in [(None, "an unknown format"), (99, "format 99")]:
        client.update_collection(name, metadata={"langgraph_store_format": stamp})
        with pytest.raises(ValueError, match=message):
            QdrantStore(client, collection_name=name).setup()


def test_invalid_index_config(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="Unsupported distance"):
        QdrantStore(
            store.client,
            index={"dims": 4, "embed": CharacterEmbeddings(4), "distance": "l2"},  # type: ignore[typeddict-item]
        )


def test_ttl_expiry_and_sweep(make_store: Any, clock: Clock) -> None:
    store = make_store(ttl={"default_ttl": 1})
    store.put(("ns",), "short", {"v": 1})
    store.put(("ns",), "forever", {"v": 2}, ttl=None)
    payload = store.client.retrieve(
        store.collection_name, [point_id(("ns",), "short")]
    )[0].payload
    assert payload["ttl_minutes"] == 1

    clock.advance(61)
    # Without omit_expired, expired items stay visible until swept.
    assert store.get(("ns",), "short", refresh_ttl=False) is not None
    assert store.sweep_ttl() == 1
    assert store.get(("ns",), "short") is None
    assert store.get(("ns",), "forever") is not None
    assert store.sweep_ttl() == 0


def test_ttl_refresh_on_read(make_store: Any, clock: Clock) -> None:
    store = make_store(ttl={"refresh_on_read": True, "omit_expired": True})
    for key in ("refreshed", "not_refreshed", "searched"):
        store.put(("ns",), key, {"key": key}, ttl=1)

    for _ in range(3):
        clock.advance(40)
        assert store.get(("ns",), "refreshed") is not None
        assert store.search(("ns",), filter={"key": "searched"})
    assert store.get(("ns",), "not_refreshed", refresh_ttl=False) is None
    assert store.get(("ns",), "refreshed") is not None
    assert store.get(("ns",), "searched") is not None


def test_omit_expired_hides_expired_items(make_store: Any, clock: Clock) -> None:
    store = make_store(ttl={"omit_expired": True, "refresh_on_read": False})
    store.put(("live",), "a", {"v": 1})
    store.put(("dead",), "b", {"v": 2}, ttl=1)
    store.put(("live",), "c", {"v": 3}, ttl=10)
    clock.advance(61)
    assert store.get(("dead",), "b") is None
    assert sorted(r.key for r in store.search(())) == ["a", "c"]
    assert store.list_namespaces() == [("live",)]


def test_ttl_refresh_tolerates_concurrently_deleted_points(make_store: Any) -> None:
    store = make_store(ttl={"refresh_on_read": True})
    store.put(("ns",), "k", {}, ttl=10)
    ops = store._refresh_operations(
        {point_id(("ns",), "k"): 10.0, point_id(("ns",), "gone"): 10.0},
        datetime.now(timezone.utc),
    )
    store.client.batch_update_points(store.collection_name, ops, wait=True)
    assert store.get(("ns",), "k") is not None


def test_ttl_sweeper_stops_when_store_is_collected(make_store: Any) -> None:
    store = make_store(ttl={"sweep_interval_minutes": 1})
    future = store.start_ttl_sweeper(sweep_interval_minutes=1 / 60 / 20)
    del store
    gc.collect()
    future.result(timeout=5)


def test_ttl_sweeper_thread(make_store: Any) -> None:
    store = make_store(ttl={"sweep_interval_minutes": 1})
    store.put(("ns",), "k", {}, ttl=1 / 60 / 10)
    future = store.start_ttl_sweeper(sweep_interval_minutes=1 / 60 / 10)
    try:
        deadline = time.monotonic() + 5
        while store.client.count(store.collection_name).count:
            assert time.monotonic() < deadline, "sweeper did not delete the item"
            time.sleep(0.05)
    finally:
        assert store.stop_ttl_sweeper(timeout=5)
    future.result(timeout=5)
