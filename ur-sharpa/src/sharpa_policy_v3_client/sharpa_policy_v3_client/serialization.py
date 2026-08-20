"""Binary msgpack helpers for the SharpA policy v3 wire contract."""

from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np


MAX_MESSAGE_SIZE = 64 * 1024 * 1024


class SerializationError(ValueError):
    """The payload cannot be represented by the v3 msgpack contract."""


class MessageTooLargeError(SerializationError):
    """The encoded or received message exceeds the configured hard limit."""


def _message_limit(max_size: int) -> int:
    if isinstance(max_size, bool) or not isinstance(max_size, int):
        raise TypeError("max_size must be an integer")
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if max_size > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"max_size cannot exceed the {MAX_MESSAGE_SIZE}-byte hard limit"
        )
    return max_size


def _encode_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        dtype = value.dtype
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            raise SerializationError(f"unsupported numpy dtype: {dtype}")
        if value.nbytes > MAX_MESSAGE_SIZE:
            raise MessageTooLargeError(
                f"numpy array is {value.nbytes} bytes; limit is {MAX_MESSAGE_SIZE}"
            )
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(order="C"),
            b"dtype": dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot msgpack-encode {type(value).__name__}")


def _descriptor_value(value: dict[Any, Any], key: bytes) -> Any:
    text_key = key.decode("ascii")
    if key in value and text_key in value:
        raise SerializationError(f"duplicate ndarray descriptor key: {text_key}")
    if key in value:
        return value[key]
    if text_key in value:
        return value[text_key]
    raise SerializationError(f"missing ndarray descriptor key: {text_key}")


def _decode_numpy(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" not in value and "__ndarray__" not in value:
        return value

    marker = _descriptor_value(value, b"__ndarray__")
    if marker is not True:
        raise SerializationError("__ndarray__ marker must be true")
    allowed = {
        b"__ndarray__",
        b"data",
        b"dtype",
        b"shape",
        "__ndarray__",
        "data",
        "dtype",
        "shape",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise SerializationError("unknown ndarray descriptor fields")

    data = _descriptor_value(value, b"data")
    if not isinstance(data, bytes):
        raise SerializationError("ndarray data must be binary")
    dtype_value = _descriptor_value(value, b"dtype")
    if isinstance(dtype_value, bytes):
        try:
            dtype_value = dtype_value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SerializationError("ndarray dtype must be ASCII") from exc
    if not isinstance(dtype_value, str):
        raise SerializationError("ndarray dtype must be a string")
    try:
        dtype = np.dtype(dtype_value)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"invalid ndarray dtype: {dtype_value!r}") from exc
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise SerializationError(f"unsupported numpy dtype: {dtype}")

    shape_value = _descriptor_value(value, b"shape")
    if not isinstance(shape_value, (list, tuple)):
        raise SerializationError("ndarray shape must be an array")
    shape: list[int] = []
    for dimension in shape_value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise SerializationError("ndarray dimensions must be integers")
        if dimension < 0:
            raise SerializationError("ndarray dimensions must be non-negative")
        shape.append(dimension)
    expected_size = math.prod(shape) * dtype.itemsize
    if expected_size != len(data):
        raise SerializationError(
            "ndarray byte length mismatch: "
            f"expected {expected_size}, received {len(data)}"
        )
    try:
        return np.frombuffer(data, dtype=dtype).reshape(tuple(shape)).copy(order="C")
    except (TypeError, ValueError) as exc:
        raise SerializationError("invalid ndarray descriptor") from exc


def packb(payload: Any, *, max_size: int = MAX_MESSAGE_SIZE) -> bytes:
    """Encode a payload and enforce the v3 message-size hard limit."""

    limit = _message_limit(max_size)
    try:
        encoded = msgpack.packb(
            payload,
            default=_encode_numpy,
            use_bin_type=True,
            strict_types=False,
        )
    except (SerializationError, MessageTooLargeError):
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SerializationError(f"msgpack encoding failed: {exc}") from exc
    if len(encoded) > limit:
        raise MessageTooLargeError(
            f"encoded message is {len(encoded)} bytes; limit is {limit}"
        )
    return encoded


def unpackb(payload: bytes | bytearray | memoryview, *, max_size: int = MAX_MESSAGE_SIZE) -> Any:
    """Decode one complete binary msgpack payload with strict ndarray handling."""

    limit = _message_limit(max_size)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    if len(payload) > limit:
        raise MessageTooLargeError(
            f"received message is {len(payload)} bytes; limit is {limit}"
        )
    try:
        return msgpack.unpackb(
            payload,
            object_hook=_decode_numpy,
            raw=False,
            strict_map_key=False,
            max_bin_len=limit,
            max_str_len=limit,
            max_array_len=limit,
            max_map_len=limit,
            max_ext_len=limit,
        )
    except (SerializationError, MessageTooLargeError):
        raise
    except (msgpack.exceptions.UnpackException, ValueError, TypeError, OverflowError) as exc:
        raise SerializationError(f"msgpack decoding failed: {exc}") from exc
