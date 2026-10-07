"""Qdrant-backed LangGraph `BaseStore`.

Each item is one point whose ID is a uuid5 of `(namespace, key)`. With an index
config, every embedded text of an item is a sub-vector of one `MAX_SIM`
multivector, so a query scores each item by its best-matching field.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import json
import logging
import math
import re
import threading
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from operator import itemgetter
from typing import Any, Literal, NamedTuple, TypeVar, cast
from urllib.parse import urlsplit

import httpx
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
)
from qdrant_client import QdrantClient, models

from langgraph.store.qdrant import _filters as filters
from langgraph.store.qdrant._payload import (
    build_payload,
    expires_at,
    is_expired,
    item_fields,
    normalize_value,
    parse_dt,
    point_id,
)
from langgraph.store.qdrant._plans import (
    Call,
    EmbedDocuments,
    EmbedQueries,
    Gather,
    Plan,
    Request,
    call,
    gather,
    run,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_COLLECTION_NAME = "langgraph_store"
VECTOR_NAME = "embedding"

MAX_ITEM_BYTES = 30 * 1024 * 1024
"""Qdrant rejects requests over 32 MiB by default."""

MIN_SERVER_VERSION = (1, 16, 0)
"""Older servers cannot store collection metadata, which holds `FORMAT_VERSION`."""

FORMAT_VERSION = 1
"""How items are laid out in Qdrant; recorded in the collection's metadata."""
_FORMAT_KEY = "langgraph_store_format"

_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_POINTS_PER_REQUEST = 512
_MAX_IDS_PER_REQUEST = 1024
_MAX_PROBES_PER_REQUEST = 256
# Facets cost nothing extra for a limit larger than their result.
_FACET_LIMIT = 2**31 - 1
# Upper bound of the JSON size of one vector component.
_BYTES_PER_FLOAT = 24

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

DistanceType = Literal["cosine", "dot", "euclid", "manhattan"]


class _Metric(NamedTuple):
    distance: models.Distance
    sign: int


# Qdrant returns euclid and manhattan as distances; negating them makes higher better.
_METRICS: dict[str, _Metric] = {
    "cosine": _Metric(models.Distance.COSINE, 1),
    "dot": _Metric(models.Distance.DOT, 1),
    "euclid": _Metric(models.Distance.EUCLID, -1),
    "manhattan": _Metric(models.Distance.MANHATTAN, -1),
}


def _keyword(**options: Any) -> models.KeywordIndexParams:
    return models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, **options)


# Qdrant's multitenancy layout: only `ns_prefixes`, which every search filters on,
# gets HNSW subgraphs, and `is_tenant` stores each namespace's points together.
_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaParams] = {
    "ns_prefixes": _keyword(),
    "ns_path": _keyword(is_tenant=True, enable_hnsw=False),
    "ns_suffixes": _keyword(enable_hnsw=False),
    "value_paths": _keyword(enable_hnsw=False),
    "ns_depth": models.IntegerIndexParams(
        type=models.IntegerIndexType.INTEGER, lookup=False, enable_hnsw=False
    ),
    **{
        field: models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME, enable_hnsw=False
        )
        for field in ("updated_at", "expires_at")
    },
}

_ITEM_FIELDS = [
    "namespace",
    "key",
    "value_json",
    "created_at",
    "updated_at",
    "expires_at",
    "ttl_minutes",
]

_NEWEST_FIRST = models.OrderByQuery(
    order_by=models.OrderBy(key="updated_at", direction=models.Direction.DESC)
)


class QdrantIndexConfig(IndexConfig, total=False):
    """`IndexConfig` (`dims`, `embed`, `fields`) plus Qdrant options."""

    distance: DistanceType
    """Defaults to `"cosine"`. For `"euclid"` and `"manhattan"` the score is the
    negated distance."""
    on_disk: bool
    hnsw_config: models.HnswConfigDiff
    quantization_config: models.QuantizationConfig
    search_params: models.SearchParams
    """For vector queries, e.g. `models.SearchParams(exact=True)` or
    `models.SearchParams(hnsw_ef=256)`."""


@dataclass
class _Upsert:
    pid: str
    ttl: float | None
    payload: dict[str, Any]
    request_bytes: int
    text_ids: list[int]


@dataclass
class _PutPlan:
    upserts: list[_Upsert] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _chunks(items: Sequence[_T], size: int) -> list[Sequence[_T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _pack(
    items: Iterable[_T], size: Callable[[_T], int], max_bytes: int, max_count: int
) -> list[list[_T]]:
    """Group items into batches bounded by total size and count."""
    batches: list[list[_T]] = []
    used = 0
    for item in items:
        if (
            not batches
            or len(batches[-1]) >= max_count
            or used + size(item) > max_bytes
        ):
            batches.append([])
            used = 0
        batches[-1].append(item)
        used += size(item)
    return batches


def _check_server_version(version: str) -> None:
    found = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if found and tuple(map(int, found.groups())) < MIN_SERVER_VERSION:
        required = ".".join(map(str, MIN_SERVER_VERSION))
        raise RuntimeError(
            f"Qdrant server {version} is not supported; langgraph-store-qdrant "
            f"requires {required} or newer."
        )


def _client_kwargs(url: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    # For local servers qdrant-client opens a connection per request, which
    # fails intermittently ("Bad file descriptor") under multi-threaded use.
    pooled = url != ":memory:" and not {"limits", "pool_size"} & kwargs.keys()
    host = urlsplit(url if "://" in url else f"//{url}").hostname
    if pooled and host in _LOCAL_HOSTS:
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=32)
        return {**kwargs, "limits": limits}
    return kwargs


@contextmanager
def _quiet_local_index_warning() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        yield


class BaseQdrantStore:
    """Store logic shared by the sync and async stores, written as plans."""

    collection_name: str
    index_config: QdrantIndexConfig | None
    embeddings: Embeddings | None
    ttl_config: TTLConfig | None

    def _init_config(
        self,
        *,
        collection_name: str,
        index: QdrantIndexConfig | None,
        ttl: TTLConfig | None,
    ) -> None:
        self.collection_name = collection_name
        self.ttl_config = ttl
        self._omit_expired = bool(ttl and ttl.get("omit_expired"))
        self.index_config = None
        self.embeddings = None
        self._fields: list[str] = []
        self._metric = _METRICS["cosine"]
        if index:
            distance = index.get("distance", "cosine")
            _require(
                distance in _METRICS,
                f"Unsupported distance {distance!r}; "
                f"expected one of {sorted(_METRICS)}",
            )
            _require(index.get("dims"), "The index config requires `dims`.")
            fields = index.get("fields") or ["$"]
            self.index_config = cast(QdrantIndexConfig, dict(index))
            self.embeddings = ensure_embeddings(index.get("embed"))
            self._fields = [fields] if isinstance(fields, str) else list(fields)
            self._metric = _METRICS[distance]

    def _vectors_config(self) -> dict[str, models.VectorParams]:
        if not self.index_config:
            return {}
        config = self.index_config
        return {
            VECTOR_NAME: models.VectorParams(
                size=config["dims"],
                distance=self._metric.distance,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                on_disk=config.get("on_disk"),
                hnsw_config=config.get("hnsw_config"),
                quantization_config=config.get("quantization_config"),
            )
        }

    def _validate_collection(self, info: models.CollectionInfo) -> None:
        if not self.index_config:
            return
        name = self.collection_name
        vectors = info.config.params.vectors
        params = vectors.get(VECTOR_NAME) if isinstance(vectors, dict) else None
        if params is None:
            raise ValueError(
                f"Collection {name!r} exists but has no {VECTOR_NAME!r} vector. It "
                "was probably created without an index config; use a new "
                "collection to enable semantic search."
            )
        expected = self._vectors_config()[VECTOR_NAME]
        _require(
            (params.size, params.distance) == (expected.size, expected.distance),
            f"Collection {name!r} has vectors of size {params.size} with "
            f"{params.distance} distance, but the store is configured for size "
            f"{expected.size} with {expected.distance}.",
        )
        _require(
            params.multivector_config,
            f"Collection {name!r} has a single-vector {VECTOR_NAME!r} field; the "
            "store requires a multivector.",
        )

    def _setup_plan(self, collection_options: dict[str, Any]) -> Plan[None]:
        info = yield from call("info")
        _check_server_version(info.version)
        name = self.collection_name
        if not (yield from call("collection_exists", name)):
            try:
                yield from call(
                    "create_collection",
                    name,
                    vectors_config=self._vectors_config(),
                    metadata={_FORMAT_KEY: FORMAT_VERSION},
                    **collection_options,
                )
            except Exception:
                # Another process may have created it concurrently.
                if not (yield from call("collection_exists", name)):
                    raise
        info = yield from call("get_collection", name)
        self._validate_collection(info)
        yield from self._check_format_plan(info)
        yield from gather(
            call("create_payload_index", name, field_name, schema, wait=True)
            for field_name, schema in _PAYLOAD_INDEXES.items()
        )

    def _check_format_plan(self, info: models.CollectionInfo) -> Plan[None]:
        name = self.collection_name
        found = (info.config.metadata or {}).get(_FORMAT_KEY)
        if found == FORMAT_VERSION:
            return
        if found is None:
            points, _ = yield from call("scroll", name, limit=1, with_vectors=False)
            if not points:  # created by hand, e.g. with custom sharding
                yield from call(
                    "update_collection", name, metadata={_FORMAT_KEY: FORMAT_VERSION}
                )
                return
        raise ValueError(
            f"Collection {name!r} holds data in "
            f"{f'format {found}' if found else 'an unknown format'}, but this "
            f"version of langgraph-store-qdrant uses format {FORMAT_VERSION}. "
            "Use a new collection."
        )

    def _sweep_plan(self) -> Plan[int]:
        expired = filters.expired(_now())
        result = yield from call(
            "count", self.collection_name, count_filter=expired, exact=True
        )
        if result.count:
            yield from call(
                "delete",
                self.collection_name,
                points_selector=models.FilterSelector(filter=expired),
                wait=True,
            )
        return cast(int, result.count)

    def _batch_plan(self, ops: Iterable[Op]) -> Plan[list[Result]]:
        ops = list(ops)
        results: list[Result] = [None] * len(ops)
        refresh: dict[str, float] = {}
        now = _now()
        omit = self._omit_expired

        # Everything is validated before the first request, so an invalid
        # filter or value fails the batch without side effects.
        gets: list[tuple[int, GetOp]] = []
        searches: list[tuple[int, SearchOp, models.Filter | None]] = []
        lists: list[tuple[int, ListNamespacesOp, models.Filter | None]] = []
        puts: list[PutOp] = []
        for idx, op in enumerate(ops):
            match op:
                case GetOp():
                    gets.append((idx, op))
                case SearchOp() | ListNamespacesOp() if op.offset < 0 or op.limit < 0:
                    raise ValueError(
                        "offset and limit must be non-negative, got "
                        f"offset={op.offset}, limit={op.limit}"
                    )
                case SearchOp(limit=0):
                    results[idx] = []  # Qdrant rejects a zero limit
                case SearchOp():
                    flt = filters.search_filter(op, now=now, omit_expired=omit)
                    searches.append((idx, op, flt))
                case ListNamespacesOp():
                    flt = filters.list_namespaces_filter(op, now=now, omit_expired=omit)
                    lists.append((idx, op, flt))
                case PutOp():
                    puts.append(op)
                case _:
                    raise ValueError(f"Unknown operation type: {type(op)}")
        put_plan = self._plan_puts(puts)

        # Reads see the state before this batch's writes. Preparing upserts can
        # fail (embedding), so it also runs before the first write.
        *_, upserts = yield from gather(
            [
                self._gets_plan(gets, results, refresh, now),
                self._searches_plan(searches, results, refresh),
                *(self._list_namespaces_plan(*lst, results) for lst in lists),
                self._prepare_upserts_plan(put_plan, now),
            ]
        )

        if refresh:
            yield from call(
                "batch_update_points",
                self.collection_name,
                update_operations=self._refresh_operations(refresh, now),
                wait=True,
            )
        for chunk in _chunks(put_plan.deletes, _MAX_IDS_PER_REQUEST):
            yield from call(
                "delete",
                self.collection_name,
                points_selector=models.PointIdsList(points=list(chunk)),
                wait=True,
            )
        for points in upserts:
            yield from call("upsert", self.collection_name, points=points, wait=True)
        return results

    def _retrieve_plan(
        self, ids: Iterable[str], with_payload: bool | list[str]
    ) -> Plan[dict[str, dict[str, Any]]]:
        """Payloads by point ID; missing points are absent."""
        chunks = yield from gather(
            call(
                "retrieve",
                self.collection_name,
                ids=list(chunk),
                with_payload=with_payload,
                with_vectors=False,
            )
            for chunk in _chunks(list(dict.fromkeys(ids)), _MAX_IDS_PER_REQUEST)
        )
        return {str(r.id): r.payload or {} for chunk in chunks for r in chunk}

    @staticmethod
    def _note_refresh(
        refresh: dict[str, float], op: GetOp | SearchOp, pid: str, payload: Mapping
    ) -> None:
        if op.refresh_ttl and payload.get("ttl_minutes") is not None:
            refresh[pid] = payload["ttl_minutes"]

    def _gets_plan(
        self,
        gets: Sequence[tuple[int, GetOp]],
        results: list[Result],
        refresh: dict[str, float],
        now: datetime,
    ) -> Plan[None]:
        pids = [point_id(op.namespace, op.key) for _, op in gets]
        payloads = yield from self._retrieve_plan(pids, _ITEM_FIELDS)
        for (idx, op), pid in zip(gets, pids, strict=True):
            payload = payloads.get(pid)
            if payload is None or (self._omit_expired and is_expired(payload, now)):
                continue
            results[idx] = Item(**item_fields(payload))
            self._note_refresh(refresh, op, pid, payload)

    def _is_scored(self, op: SearchOp) -> bool:
        return bool(op.query and self.index_config)

    def _search_request(
        self,
        op: SearchOp,
        query_filter: models.Filter | None,
        vector: list[float] | None,
        *,
        limit: int,
        with_payload: bool | list[str],
    ) -> models.QueryRequest:
        query: dict[str, Any] = {"query": _NEWEST_FIRST}
        if self._is_scored(op):
            params = cast(QdrantIndexConfig, self.index_config).get("search_params")
            query = {"query": [vector], "using": VECTOR_NAME, "params": params}
        return models.QueryRequest(
            **query,
            filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vector=False,
        )

    def _window_payload(self, op: SearchOp) -> bool | list[str]:
        """Full items for a first page; deeper pages fetch only ranking fields
        and retrieve the returned page afterwards."""
        if op.offset == 0:
            return _ITEM_FIELDS
        return False if self._is_scored(op) else ["updated_at"]

    def _rank(self, op: SearchOp, point: models.ScoredPoint) -> Any:
        """Sort key, higher is better: the score, or `updated_at`."""
        if self._is_scored(op):
            return self._metric.sign * point.score
        return parse_dt((point.payload or {})["updated_at"])

    def _searches_plan(
        self,
        searches: Sequence[tuple[int, SearchOp, models.Filter | None]],
        results: list[Result],
        refresh: dict[str, float],
    ) -> Plan[None]:
        """Searches with stable pagination.

        Qdrant orders ties (equal timestamps or scores) inconsistently across
        requests, so pages are cut client-side: each search fetches
        `offset + limit + 1` results, and a page boundary inside a group of
        ties is resolved by point ID.
        """
        if not searches:
            return
        queries = list(
            dict.fromkeys(
                op.query for _, op, _ in searches if op.query and self.index_config
            )
        )
        vectors: dict[str | None, list[float]] = {}
        if queries:
            embedded = self._check_vectors((yield EmbedQueries(queries)), len(queries))
            vectors = dict(zip(queries, embedded, strict=True))
        responses = yield from call(
            "query_batch_points",
            self.collection_name,
            requests=[
                self._search_request(
                    op,
                    flt,
                    vectors.get(op.query),
                    limit=op.offset + op.limit + 1,
                    with_payload=self._window_payload(op),
                )
                for _, op, flt in searches
            ],
        )
        payloads = {
            str(p.id): p.payload
            for response in responses
            for p in response.points
            if p.payload and "value_json" in p.payload
        }
        pages = yield from gather(
            self._page_plan(op, flt, vectors.get(op.query), response.points)
            for (_, op, flt), response in zip(searches, responses, strict=True)
        )
        missing = [pid for page in pages for _, pid in page if pid not in payloads]
        payloads |= yield from self._retrieve_plan(missing, _ITEM_FIELDS)
        for (idx, op, _), page in zip(searches, pages, strict=True):
            found = [
                (rank, pid, payloads[pid]) for rank, pid in page if pid in payloads
            ]
            results[idx] = [
                SearchItem(
                    **item_fields(payload),
                    score=float(rank) if self._is_scored(op) else None,
                )
                for rank, _, payload in found
            ]
            for _, pid, payload in found:
                self._note_refresh(refresh, op, pid, payload)

    def _page_plan(
        self,
        op: SearchOp,
        query_filter: models.Filter | None,
        vector: list[float] | None,
        points: Sequence[models.ScoredPoint],
    ) -> Plan[list[tuple[Any, str]]]:
        """One page, ordered by rank descending, then point ID."""
        window = op.offset + op.limit
        ranked = [(self._rank(op, p), str(p.id)) for p in points]
        if len(points) > window and ranked[window][0] == ranked[window - 1][0]:
            # The page boundary cuts a group of ties: complete the group.
            boundary = ranked[window - 1][0]
            better = [r for r in ranked if r[0] > boundary]
            if self._is_scored(op):
                ties = yield from self._score_ties_plan(
                    op, query_filter, cast(list[float], vector), boundary
                )
            else:
                ties = yield from self._time_ties_plan(
                    query_filter, boundary, window - len(better)
                )
            ranked = better + [(boundary, pid) for pid in ties]
        by_id = sorted(ranked, key=itemgetter(1))
        return sorted(by_id, key=itemgetter(0), reverse=True)[op.offset : window]

    def _time_ties_plan(
        self, base: models.Filter | None, boundary: datetime, needed: int
    ) -> Plan[list[str]]:
        """The `needed` lowest point IDs updated exactly at `boundary`.

        Scroll returns points in ID order, so it stops after `needed` IDs
        however large the group is.
        """
        same_time = filters.within("updated_at", gte=boundary, lte=boundary)
        scroll_filter = filters.every(*([base] if base else []), same_time)
        ids: list[str] = []
        offset: models.ExtendedPointId | None = None
        while len(ids) < needed:
            points, offset = yield from call(
                "scroll",
                self.collection_name,
                scroll_filter=scroll_filter,
                limit=min(needed - len(ids), _MAX_IDS_PER_REQUEST),
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids += [str(p.id) for p in points]
            if offset is None:
                break
        return ids

    def _score_ties_plan(
        self,
        op: SearchOp,
        query_filter: models.Filter | None,
        vector: list[float],
        boundary: float,
    ) -> Plan[list[str]]:
        """IDs of all results scoring exactly `boundary`."""
        limit = 2 * (op.offset + op.limit + 1)
        while True:
            request = self._search_request(
                op, query_filter, vector, limit=limit, with_payload=False
            )
            (response,) = yield from call(
                "query_batch_points", self.collection_name, requests=[request]
            )
            points = response.points
            if len(points) < limit or self._rank(op, points[-1]) != boundary:
                return [str(p.id) for p in points if self._rank(op, p) == boundary]
            limit *= 2

    def _list_namespaces_plan(
        self,
        idx: int,
        op: ListNamespacesOp,
        base_filter: models.Filter | None,
        results: list[Result],
    ) -> Plan[None]:
        """Candidates from an approximate facet over `ns_path`, confirmed in order.

        Exact facets time out on collections with a few hundred thousand
        namespaces, while approximate ones may still report a namespace whose
        items were all deleted. Only the namespaces up to the requested page are
        confirmed, a chunk at a time.
        """
        response = yield from call(
            "facet",
            self.collection_name,
            key="ns_path",
            facet_filter=base_filter,
            limit=_FACET_LIMIT,
            exact=False,
        )
        paths = [str(hit.value) for hit in response.hits]
        conditions = op.match_conditions or ()
        candidates: dict[tuple[str, ...], list[str]] = {}
        for path, labels in zip(paths, json.loads(f"[{','.join(paths)}]"), strict=True):
            namespace = tuple(labels)
            if all(filters.namespace_matches(namespace, c) for c in conditions):
                candidates.setdefault(namespace[: op.max_depth], []).append(path)
        wanted = op.offset + op.limit
        listed: list[tuple[str, ...]] = []
        for chunk in _chunks(sorted(candidates), _MAX_PROBES_PER_REQUEST):
            if len(listed) >= wanted:
                break
            live = yield from self._live_plan(
                {namespace: candidates[namespace] for namespace in chunk}, base_filter
            )
            listed += [namespace for namespace in chunk if namespace in live]
        results[idx] = listed[op.offset : wanted]

    def _live_plan(
        self,
        candidates: Mapping[tuple[str, ...], list[str]],
        base_filter: models.Filter | None,
    ) -> Plan[set[tuple[str, ...]]]:
        """The namespaces with at least one candidate `ns_path` holding items.

        Each round checks one more candidate per namespace with an exact facet.
        """
        pending = {namespace: list(paths) for namespace, paths in candidates.items()}
        live: set[tuple[str, ...]] = set()
        while pending:
            probes = {paths.pop(): namespace for namespace, paths in pending.items()}
            response = yield from call(
                "facet",
                self.collection_name,
                key="ns_path",
                facet_filter=filters.every(
                    *([base_filter] if base_filter else []),
                    filters.match_any("ns_path", list(probes)),
                ),
                limit=len(probes),
                exact=True,
            )
            live |= {probes[str(hit.value)] for hit in response.hits if hit.count}
            pending = {
                ns: paths for ns, paths in pending.items() if ns not in live and paths
            }
        return live

    def _plan_puts(self, ops: Sequence[PutOp]) -> _PutPlan:
        deduped = {(tuple(op.namespace), op.key): op for op in ops}
        plan = _PutPlan()
        text_ids: dict[str, int] = {}
        for (namespace, key), op in deduped.items():
            if op.value is None:
                plan.deletes.append(point_id(namespace, key))
            else:
                plan.upserts.append(self._upsert(namespace, key, op, text_ids))
        plan.texts = list(text_ids)
        return plan

    def _upsert(
        self, namespace: tuple[str, ...], key: str, op: PutOp, text_ids: dict[str, int]
    ) -> _Upsert:
        value = normalize_value(cast(Mapping[str, Any], op.value))
        payload = build_payload(namespace, key, value, op.ttl)
        ids = [
            text_ids.setdefault(text, len(text_ids))
            for text in self._texts_to_embed(op, value)
        ]
        dims = self.index_config["dims"] if self.index_config else 0
        # Escaped JSON bounds the request size; 256 bytes cover the timestamps
        # and the point envelope.
        size = len(json.dumps(payload)) + 256 + _BYTES_PER_FLOAT * dims * len(ids)
        _require(
            size <= MAX_ITEM_BYTES,
            f"Item {key!r} in {namespace} is about {size} bytes when serialized, "
            f"more than the {MAX_ITEM_BYTES} bytes a single Qdrant request can carry.",
        )
        return _Upsert(point_id(namespace, key), op.ttl, payload, size, ids)

    def _check_vectors(
        self, vectors: Sequence[Sequence[float]], expected: int
    ) -> list[list[float]]:
        dims = cast(QdrantIndexConfig, self.index_config)["dims"]
        _require(
            len(vectors) == expected,
            f"The embedding function returned {len(vectors)} vectors for "
            f"{expected} texts.",
        )
        sizes = {len(vector) for vector in vectors} - {dims}
        _require(
            not sizes,
            f"The embedding function returned a vector of size {min(sizes or [0])}, "
            f"but the index is configured for {dims} dims.",
        )
        _require(
            all(math.isfinite(x) for vector in vectors for x in vector),
            "The embedding function returned a NaN or infinite value.",
        )
        return [list(vector) for vector in vectors]

    def _texts_to_embed(self, op: PutOp, value: dict[str, Any]) -> list[str]:
        if not self.index_config or op.index is False:
            return []
        paths = self._fields if op.index is None else op.index
        return [text for path in paths for text in get_text_at_path(value, path)]

    def _prepare_upserts_plan(
        self, plan: _PutPlan, now: datetime
    ) -> Plan[list[list[models.PointStruct]]]:
        """Points to upsert, in size-bounded requests."""
        if not plan.upserts:
            return []
        existing = yield from self._retrieve_plan(
            (u.pid for u in plan.upserts), ["created_at"]
        )
        vectors: list[list[float]] = []
        if plan.texts:
            embedded = yield EmbedDocuments(plan.texts)
            vectors = self._check_vectors(embedded, len(plan.texts))
        updated_at = now.isoformat()

        def point(upsert: _Upsert) -> models.PointStruct:
            created_at = existing.get(upsert.pid, {}).get("created_at")
            vector: dict[str, Any] = {
                VECTOR_NAME: [vectors[i] for i in upsert.text_ids]
            }
            return models.PointStruct(
                id=upsert.pid,
                vector=vector if upsert.text_ids else {},
                payload={
                    **upsert.payload,
                    "created_at": created_at or updated_at,
                    "updated_at": updated_at,
                    "expires_at": expires_at(now, upsert.ttl),
                },
            )

        batches = _pack(
            plan.upserts,
            size=lambda u: u.request_bytes,
            max_bytes=_MAX_REQUEST_BYTES,
            max_count=_MAX_POINTS_PER_REQUEST,
        )
        return [[point(upsert) for upsert in batch] for batch in batches]

    def _refresh_operations(
        self, refresh: Mapping[str, float], now: datetime
    ) -> list[models.UpdateOperation]:
        by_ttl = itertools.groupby(
            sorted(refresh.items(), key=itemgetter(1)), itemgetter(1)
        )
        return [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload={"expires_at": expires_at(now, ttl)},
                    # A filter rather than IDs: refreshing a point deleted
                    # concurrently is then a no-op instead of an error.
                    filter=filters.every(
                        models.HasIdCondition(has_id=[pid for pid, _ in group])
                    ),
                )
            )
            for ttl, group in by_ttl
        ]


def _sweep_interval_seconds(
    ttl_config: TTLConfig, sweep_interval_minutes: float | None
) -> float:
    minutes = sweep_interval_minutes or ttl_config.get("sweep_interval_minutes") or 5
    return float(minutes) * 60


def _sweep_forever(
    ref: weakref.ReferenceType[QdrantStore],
    stop: threading.Event,
    interval: float,
    future: concurrent.futures.Future[None],
) -> None:
    while not stop.wait(interval):
        store = ref()
        if store is None:
            break
        try:
            if expired := store.sweep_ttl():
                logger.info("Store swept %d expired items", expired)
        except Exception:
            logger.exception("Store TTL sweep iteration failed")
        del store
    if not future.cancelled():
        future.set_result(None)


class QdrantStore(BaseStore, BaseQdrantStore):
    """Long-term memory store backed by Qdrant.

    Call `setup()` once before first use.

    Examples:
        ```python
        from qdrant_client import QdrantClient
        from langgraph.store.qdrant import QdrantStore

        store = QdrantStore(QdrantClient(url="http://localhost:6333"))
        store.setup()
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        ```

        Semantic search:

        ```python
        from langchain.embeddings import init_embeddings

        with QdrantStore.from_url(
            "http://localhost:6333",
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"],
            },
        ) as store:
            store.setup()
            store.put(("docs",), "doc1", {"text": "Python tutorial"})
            results = store.search(("docs",), query="python programming")
        ```
    """

    supports_ttl = True

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        index: QdrantIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        """Without an `index` config, `query` arguments to `search` are ignored and
        no vectors are stored."""
        super().__init__()
        self.client = client
        self._init_config(
            collection_name=collection_name,
            index=index,
            ttl=ttl,
        )
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_sweeper_future: concurrent.futures.Future[None] | None = None
        self._ttl_stop_event = threading.Event()

    @classmethod
    @contextmanager
    def from_url(
        cls,
        url: str,
        *,
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        index: QdrantIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        **client_kwargs: Any,
    ) -> Iterator[QdrantStore]:
        """A store with its own client, closed on exit.

        `url` is a Qdrant URL or `":memory:"`; extra keyword arguments go to
        `QdrantClient`.
        """
        client = QdrantClient(
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
            client.close()

    def _run(self, plan: Plan[_T]) -> _T:
        return run(plan, self._execute)

    def _execute(self, request: Request) -> Any:
        embeddings = cast(Embeddings, self.embeddings)
        match request:
            case Call(method, args, kwargs):
                return getattr(self.client, method)(*args, **kwargs)
            case Gather(plans):
                return [self._run(plan) for plan in plans]
            case EmbedDocuments(texts):
                return embeddings.embed_documents(texts)
            case EmbedQueries(queries):
                return [embeddings.embed_query(query) for query in queries]

    def setup(self, **collection_options: Any) -> None:
        """Create the collection and payload indexes if they do not exist.

        Keyword arguments go to Qdrant's `create_collection` when the collection
        is created, e.g. `shard_number`, `replication_factor` and
        `write_consistency_factor`.

        Raises `ValueError` for an incompatible existing collection and
        `RuntimeError` for a server older than `MIN_SERVER_VERSION`.
        """
        with _quiet_local_index_warning():
            self._run(self._setup_plan(collection_options))

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._run(self._batch_plan(ops))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.batch, list(ops)
        )

    def sweep_ttl(self) -> int:
        """Delete expired items and return how many there were."""
        return self._run(self._sweep_plan())

    def start_ttl_sweeper(
        self, sweep_interval_minutes: float | None = None
    ) -> concurrent.futures.Future[None]:
        """Delete expired items periodically in a background thread.

        The thread holds only a weak reference to the store and stops when the
        store is garbage collected. Cancelling the returned future stops it.
        """
        if not self.ttl_config:
            done: concurrent.futures.Future[None] = concurrent.futures.Future()
            done.set_result(None)
            return done
        thread = self._ttl_sweeper_thread
        if thread is not None and thread.is_alive():
            return cast(concurrent.futures.Future, self._ttl_sweeper_future)
        stop = self._ttl_stop_event = threading.Event()
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.add_done_callback(lambda f: stop.set() if f.cancelled() else None)
        interval = _sweep_interval_seconds(self.ttl_config, sweep_interval_minutes)
        thread = threading.Thread(
            target=_sweep_forever,
            args=(weakref.ref(self), stop, interval, future),
            daemon=True,
            name="ttl-sweeper",
        )
        self._ttl_sweeper_thread = thread
        self._ttl_sweeper_future = future
        thread.start()
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the sweeper thread.

        Returns `False` if a sweep in progress outlasts `timeout`; the thread
        stops once it is done.
        """
        thread = self._ttl_sweeper_thread
        if thread is not None and thread.is_alive():
            self._ttl_stop_event.set()
            thread.join(timeout)
            if thread.is_alive():
                return False
        self._ttl_sweeper_thread = None
        return True

    def __del__(self) -> None:
        if stop := getattr(self, "_ttl_stop_event", None):
            stop.set()
