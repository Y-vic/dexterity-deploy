import numpy as np
import pytest

from sharpa_policy_v3_client.hardware_geometry import (
    UrSharpAWireGeometry,
    column_pose9_to_transform,
    matrix_to_rotvec,
    relative_wire_pose_to_absolute,
    rotvec_to_matrix,
    transform_to_column_pose9,
)


def transform(position=(0.2, 0.1, 0.3), rotvec=(0.2, -0.1, 0.3)):
    value = np.eye(4)
    value[:3, :3] = rotvec_to_matrix(rotvec)
    value[:3, 3] = position
    return value


@pytest.mark.parametrize("side", ["left", "right"])
def test_capture_wire_and_model_round_trips(side):
    geometry = UrSharpAWireGeometry()
    source = transform()

    wire = geometry.capture_tcp_to_wire_pose(source, side)
    recovered = geometry.wire_pose_to_capture_tcp(wire, side)
    model_direct = geometry.capture_tcp_to_model_pose(source, side)
    model_via_wire = geometry.wire_pose_to_model_pose(wire, side)

    assert wire.dtype == np.float32
    np.testing.assert_allclose(recovered, source, atol=2.0e-6)
    np.testing.assert_allclose(model_via_wire, model_direct, atol=2.0e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ur_base_and_capture_round_trip(side):
    geometry = UrSharpAWireGeometry()
    source = transform(position=(0.4, -0.2, 0.5))

    capture = geometry.ur_base_tcp_to_capture_tcp(source, side)
    recovered = geometry.capture_tcp_to_ur_base_tcp(capture, side)

    np.testing.assert_allclose(recovered, source, atol=2.0e-6)


@pytest.mark.parametrize(
    "rotvec",
    [(0.0, 0.0, 0.0), (0.2, -0.3, 0.4), (np.pi - 1.0e-7, 0.0, 0.0)],
)
def test_rotation_vector_round_trip(rotvec):
    rotation = rotvec_to_matrix(rotvec)
    recovered = rotvec_to_matrix(matrix_to_rotvec(rotation))

    np.testing.assert_allclose(recovered, rotation, atol=2.0e-6)


def test_relative_wire_pose_is_composed_in_local_wrist_frame():
    reference = transform_to_column_pose9(
        transform(position=(0.2, 0.1, 0.3), rotvec=(0.0, 0.0, np.pi / 2.0))
    )
    relative = transform_to_column_pose9(
        transform(position=(0.01, 0.0, 0.0), rotvec=(0.0, 0.0, 0.0))
    )

    absolute = relative_wire_pose_to_absolute(relative, reference)
    absolute_transform = column_pose9_to_transform(absolute)

    np.testing.assert_allclose(absolute_transform[:3, 3], (0.2, 0.11, 0.3), atol=1.0e-6)


def test_rejects_row_rot6d_when_column_axes_are_collinear():
    bad = np.zeros(9, dtype=np.float32)
    bad[3:6] = (1.0, 0.0, 0.0)
    bad[6:9] = (2.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="degenerate"):
        column_pose9_to_transform(bad)
