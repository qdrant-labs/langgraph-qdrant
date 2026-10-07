"""Compiling store filters into Qdrant conditions.

Qdrant filters cannot tell an object from a scalar or a missing field from
`[]`, and match scalars against any element of an array. Each item's
`value_paths` markers (see `_payload.hash_value`) make those comparisons exact.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langgraph.store.base import ListNamespacesOp, MatchCondition, SearchOp
from qdrant_client import models

from langgraph.store.qdrant._payload import (
    MAX_INT,
    MIN_INT,
    dumps,
    encode_ns,
    hash_value,
    number_key,
    parse_dt,
)

Condition = models.Condition

# Qdrant compares ranges as f64, which is exact only below this magnitude.
_SAFE_FLOAT = 2**53

_SIMPLE_KEY = re.compile(r"[A-Za-z0-9_\-]+")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_INTEGER = re.compile(r"[+-]?\d+")

_AFFIX_FIELDS = {"prefix": "ns_prefixes", "suffix": "ns_suffixes"}


def match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def match_any(key: str, values: list[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def within(key: str, **bounds: Any) -> models.FieldCondition:
    dated = any(isinstance(bound, datetime) for bound in bounds.values())
    range_type = models.DatetimeRange if dated else models.Range
    return models.FieldCondition(key=key, range=range_type(**bounds))


def is_empty(key: str) -> models.IsEmptyCondition:
    return models.IsEmptyCondition(is_empty=models.PayloadField(key=key))


def every(*conditions: Condition) -> models.Filter:
    return models.Filter(must=list(conditions))


def either(*conditions: Condition) -> models.Filter:
    return models.Filter(should=list(conditions))


def negate(*conditions: Condition) -> models.Filter:
    return models.Filter(must_not=list(conditions))


def where(*conditions: Condition) -> models.Filter | None:
    return every(*conditions) if conditions else None


def not_expired(now: datetime) -> models.Filter:
    return either(is_empty("expires_at"), within("expires_at", gt=now))


def expired(now: datetime) -> models.Filter:
    return every(within("expires_at", lte=now))


@dataclass(frozen=True)
class FieldPath:
    parts: tuple[str, ...]

    @property
    def key(self) -> str:
        if any(c in part for part in self.parts for c in '"\\'):
            raise ValueError(
                f"Filter keys cannot contain double quotes or backslashes: {self.parts}"
            )
        quoted = (p if _SIMPLE_KEY.fullmatch(p) else f'"{p}"' for p in self.parts)
        return ".".join(["value", *quoted])

    def child(self, name: str) -> FieldPath:
        return FieldPath((*self.parts, name))

    def has(self, marker: str) -> models.FieldCondition:
        return match("value_paths", f"{marker}:{encode_ns(self.parts)}")

    def equals_json(self, value: Any) -> models.FieldCondition:
        """Exact equality of an object or list, via its stored content hash."""
        try:
            normalized = json.loads(dumps(value))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"Filter values must be valid JSON: {exc}") from exc
        encoded = encode_ns(self.parts)
        return match("value_paths", f"h:{encoded}:{hash_value(normalized).hex()}")

    def equals_number(self, number: float) -> Condition:
        """Exact numeric equality, `1 == 1.0`: stored integers by value, stored
        floats by their `n:` marker (Qdrant's own floats can be off by a ULP)."""
        marker = match("value_paths", f"n:{encode_ns(self.parts)}:{number_key(number)}")
        integral = isinstance(number, int) or number.is_integer()
        if integral and MIN_INT <= int(number) <= MAX_INT:
            return either(match(self.key, int(number)), marker)
        return marker

    def scalar(self, condition: Condition) -> list[Condition]:
        # Qdrant matches scalar conditions against any element of an array.
        return [condition, negate(self.has("a"))]


def equals(path: FieldPath, value: Any, *, exact: bool = False) -> list[Condition]:
    """Conditions for `<value at path> == value`.

    A plain object filter matches objects containing its keys; `exact`
    (`$eq`/`$ne`) requires the whole object to be equal.
    """
    match value:
        case Mapping() if exact:
            return [path.equals_json(dict(value))]
        case Mapping():
            nested = (field_conditions(path.child(str(k)), v) for k, v in value.items())
            return [path.has("o"), *itertools.chain.from_iterable(nested)]
        case list() | tuple():
            return [path.equals_json(list(value))]
        case None:  # missing or null; `[]` is an array
            return path.scalar(is_empty(path.key))
        case bool() | str():
            return path.scalar(match(path.key, value))
        case float() if not math.isfinite(value):
            raise ValueError(f"Filter values must be finite numbers, got {value!r}")
        case int() | float():
            return path.scalar(path.equals_number(value))
    raise ValueError(f"Unsupported filter value type: {type(value).__name__}")


def range_bound(op: str, value: Any) -> float | datetime:
    match value:
        case datetime():
            return value
        case bool():
            pass
        case float() if not math.isfinite(value):
            pass
        case int() | float() if _exact_bound(value):
            return value
        case int() | float():
            raise ValueError(
                f"Operator {op} cannot compare {value} exactly: Qdrant rounds it. "
                "Integer bounds up to ±900719925474099 are always exact."
            )
        case str() if _INTEGER.fullmatch(text := value.strip()):
            return range_bound(op, int(text))
        case str() if _NUMBER.fullmatch(text := value.strip()):
            return range_bound(op, float(text))
        case str():
            with suppress(ValueError):
                return parse_dt(value.strip())
    raise ValueError(
        f"Operator {op} requires a finite number or an ISO-8601 datetime, got {value!r}"
    )


def _exact_bound(number: float) -> bool:
    if isinstance(number, float) and abs(number) >= 2**64:
        return True  # rounding cannot reach a stored integer
    # A bound n is sent as "n.0", which Qdrant parses as (n * 10) / 10 in
    # floating point: exact only if n * 10 is.
    if abs(number) >= _SAFE_FLOAT:
        return False
    if isinstance(number, float) and not number.is_integer():
        return True
    return float(int(number) * 10) == int(number) * 10


def _comparison(bound: str) -> Callable[[FieldPath, Any], list[Condition]]:
    def compare(path: FieldPath, operand: Any) -> list[Condition]:
        limit = range_bound(f"${bound}", operand)
        return path.scalar(within(path.key, **{bound: limit}))

    return compare


OPERATORS: dict[str, Callable[[FieldPath, Any], list[Condition]]] = {
    "$eq": lambda path, operand: equals(path, operand, exact=True),
    "$ne": lambda path, operand: [negate(every(*equals(path, operand, exact=True)))],
    **{f"${bound}": _comparison(bound) for bound in ("gt", "gte", "lt", "lte")},
}


def field_conditions(path: FieldPath, value: Any) -> list[Condition]:
    if not (isinstance(value, Mapping) and any(str(k).startswith("$") for k in value)):
        return equals(path, value)
    if unknown := [op for op in value if op not in OPERATORS]:
        raise ValueError(f"Unsupported operator: {unknown[0]}")
    return [c for op, operand in value.items() for c in OPERATORS[op](path, operand)]


def search_filter(
    op: SearchOp, *, now: datetime, omit_expired: bool
) -> models.Filter | None:
    # Stored keys are strings: values are JSON-normalized.
    try:
        value_conditions = [
            field_conditions(FieldPath((str(key),)), value)
            for key, value in (op.filter or {}).items()
        ]
    except RecursionError:
        raise ValueError("Filter is nested too deeply.") from None
    return where(
        *_present(
            op.namespace_prefix, match("ns_prefixes", encode_ns(op.namespace_prefix))
        ),
        *itertools.chain.from_iterable(value_conditions),
        *_present(omit_expired, not_expired(now)),
    )


def _present(flag: object, condition: Condition) -> list[Condition]:
    return [condition] if flag else []


def _oriented(labels: Sequence[str], match_type: str) -> tuple[str, ...]:
    """Labels read from the end that `match_type` anchors on."""
    return tuple(labels) if match_type == "prefix" else tuple(reversed(labels))


def _fixed_affix(condition: MatchCondition) -> tuple[str, ...]:
    """The part of the condition path before its first wildcard."""
    oriented = _oriented(condition.path, condition.match_type)
    fixed = itertools.takewhile(lambda label: label != "*", oriented)
    return _oriented(tuple(fixed), condition.match_type)


def list_namespaces_filter(
    op: ListNamespacesOp, *, now: datetime, omit_expired: bool
) -> models.Filter | None:
    """Server-side pre-filter; wildcards are resolved by `namespace_matches`."""
    conditions = [c for c in op.match_conditions or () if c.path]
    return where(
        *(within("ns_depth", gte=len(c.path)) for c in conditions),
        *(
            match(_AFFIX_FIELDS[c.match_type], encode_ns(fixed))
            for c in conditions
            if (fixed := _fixed_affix(c))
        ),
        *_present(omit_expired, not_expired(now)),
    )


def namespace_matches(namespace: Iterable[str], condition: MatchCondition) -> bool:
    path = _oriented(condition.path, condition.match_type)
    labels = _oriented(tuple(namespace), condition.match_type)
    return len(path) <= len(labels) and all(
        p in ("*", label) for p, label in zip(path, labels, strict=False)
    )
