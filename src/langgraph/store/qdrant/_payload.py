from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, cast

MAX_VALUE_DEPTH = 32
"""qdrant-client's gRPC transport cannot encode objects nested any deeper."""

MIN_INT, MAX_INT = -(2**63), 2**63 - 1

_POINT_ID_NAMESPACE = uuid.UUID("5b2f8d0e-6c1a-4d0b-9a57-1f1f6a2c3e01")

dumps = json.JSONEncoder(
    ensure_ascii=False, allow_nan=False, separators=(",", ":")
).encode


def point_id(namespace: Sequence[str], key: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, dumps([list(namespace), key])))


def encode_ns(labels: Sequence[str]) -> str:
    return dumps(list(labels))


def parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def expires_at(updated_at: datetime, ttl: float | None) -> str | None:
    return None if ttl is None else (updated_at + timedelta(minutes=ttl)).isoformat()


def normalize_value(value: Mapping[str, Any]) -> dict[str, Any]:
    """The value exactly as stored and read back (a strict JSON round trip).

    Rejects what Qdrant would store lossily or refuse.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"Store values must be mappings, got {type(value).__name__}")
    try:
        normalized = json.loads(dumps(dict(value)))
    except ValueError as exc:
        raise ValueError(f"Store values must be valid JSON: {exc}") from exc
    except RecursionError:
        raise ValueError(_TOO_DEEP) from None
    _check_storable(normalized, depth=1)
    return cast(dict[str, Any], normalized)


_TOO_DEEP = f"Store values cannot be nested deeper than {MAX_VALUE_DEPTH} levels."


def _check_storable(value: Any, depth: int) -> None:
    match value:
        case dict() | list() if depth > MAX_VALUE_DEPTH:
            raise ValueError(_TOO_DEEP)
        case dict():
            for child in value.values():
                _check_storable(child, depth + 1)
        case list():
            for child in value:
                _check_storable(child, depth + 1)
        case int() if not MIN_INT <= value <= MAX_INT:
            raise ValueError(
                f"Integer {value} is outside the 64-bit signed range Qdrant can "
                "store and match exactly; store it as a string instead."
            )


def number_key(number: float) -> str:
    """Canonical text of a number, the same for `1` and `1.0`."""
    if isinstance(number, float) and not number.is_integer():
        return repr(number)
    return str(int(number))


def _digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


def hash_value(
    value: Any, path: tuple[str, ...] = (), markers: list[str] | None = None
) -> bytes:
    """Content hash where `1 == 1.0` and key order does not matter.

    Appends `value_paths` markers to `markers`: `o:<path>` (object), `a:<path>`
    (array) and `h:<path>:<hash>` for nested objects and arrays, and
    `n:<path>:<number_key>` for floats.
    """
    match value:
        case dict():
            kind = "o"
            data = b",".join(
                dumps(k).encode() + b":" + hash_value(value[k], (*path, k), markers)
                for k in sorted(value)
            )
        case list():
            kind = "a"
            # Lists are compared as a whole, so their elements get no markers.
            data = b",".join(hash_value(v) for v in value)
        case float():
            if markers is not None and path:
                markers.append(f"n:{encode_ns(path)}:{number_key(value)}")
            return _digest(b"s" + number_key(value).encode())
        case _:
            return _digest(b"s" + dumps(value).encode())
    digest = _digest(kind.encode() + data)
    if markers is not None and path:
        encoded = encode_ns(path)
        markers += [f"{kind}:{encoded}", f"h:{encoded}:{digest.hex()}"]
    return digest


def value_paths(value: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    hash_value(value, markers=markers)
    return markers


def build_payload(
    namespace: tuple[str, ...], key: str, value: dict[str, Any], ttl: float | None
) -> dict[str, Any]:
    """The payload without timestamps, which are set when writing."""
    depths = range(1, len(namespace) + 1)
    return {
        "namespace": list(namespace),
        "ns_path": encode_ns(namespace),
        "ns_prefixes": [encode_ns(namespace[:d]) for d in depths],
        "ns_suffixes": [encode_ns(namespace[-d:]) for d in depths],
        "ns_depth": len(namespace),
        "key": key,
        "value": value,
        # Qdrant can change floats by one unit in the last place and reorder
        # keys, so items are read from this exact copy.
        "value_json": dumps(value),
        "value_paths": value_paths(value),
        "ttl_minutes": ttl,
    }


def item_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "namespace": tuple(payload["namespace"]),
        "key": payload["key"],
        "value": json.loads(payload["value_json"]),
        "created_at": parse_dt(payload["created_at"]),
        "updated_at": parse_dt(payload["updated_at"]),
    }


def is_expired(payload: Mapping[str, Any], now: datetime) -> bool:
    expiry = payload.get("expires_at")
    return expiry is not None and parse_dt(expiry) <= now
