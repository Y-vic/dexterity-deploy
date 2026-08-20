from __future__ import annotations

import msgpack
import numpy as np
import pytest

from sharpa_policy_v3_client.serialization import (
    MAX_MESSAGE_SIZE,
    MessageTooLargeError,
    SerializationError,
    packb,
    unpackb,
)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (np.float32, (2, 3)),
        (np.int64, (3,)),
        (np.bool_, (2, 2)),
        (np.uint8, (0, 5)),
        (np.float64, ()),
    ],
)
def test_ndarray_round_trip_preserves_dtype_shape_and_values(dtype, shape):
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    source = np.arange(size, dtype=np.int64).astype(dtype).reshape(shape)

    decoded = unpackb(packb({"array": source}))

    result = decoded["array"]
    assert isinstance(result, np.ndarray)
    assert result.dtype == source.dtype
    assert result.shape == source.shape
    np.testing.assert_array_equal(result, source)
    assert result.flags.c_contiguous


def test_ndarray_descriptor_matches_wire_contract_exactly():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]

    wire = msgpack.unpackb(
        packb({"array": source}),
        raw=False,
        strict_map_key=False,
    )

    descriptor = wire["array"]
    assert set(descriptor) == {b"__ndarray__", b"data", b"dtype", b"shape"}
    assert descriptor[b"__ndarray__"] is True
    assert descriptor[b"data"] == source.tobytes(order="C")
    assert descriptor[b"dtype"] == source.dtype.str
    assert descriptor[b"shape"] == list(source.shape)


def test_numpy_scalars_are_encoded_as_native_msgpack_scalars():
    decoded = unpackb(packb({"count": np.int64(4), "valid": np.bool_(True)}))

    assert decoded == {"count": 4, "valid": True}
    assert type(decoded["count"]) is int
    assert type(decoded["valid"]) is bool


def test_object_arrays_are_rejected():
    with pytest.raises(SerializationError, match="unsupported numpy dtype"):
        packb(np.asarray([object()], dtype=object))


def test_malformed_ndarray_byte_length_is_rejected():
    malformed = msgpack.packb(
        {
            b"__ndarray__": True,
            b"data": b"\x00\x01",
            b"dtype": "<f4",
            b"shape": (2,),
        },
        use_bin_type=True,
    )

    with pytest.raises(SerializationError, match="byte length mismatch"):
        unpackb(malformed)


def test_encoded_and_received_messages_obey_configured_limit():
    with pytest.raises(MessageTooLargeError, match="encoded message"):
        packb({"blob": b"x" * 64}, max_size=32)

    encoded = packb({"blob": b"x" * 64})
    with pytest.raises(MessageTooLargeError, match="received message"):
        unpackb(encoded, max_size=32)


def test_configured_limit_cannot_exceed_hard_limit():
    with pytest.raises(ValueError, match="hard limit"):
        packb({}, max_size=MAX_MESSAGE_SIZE + 1)

    with pytest.raises(ValueError, match="hard limit"):
        unpackb(b"\x80", max_size=MAX_MESSAGE_SIZE + 1)
