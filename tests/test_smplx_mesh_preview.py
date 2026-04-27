"""Tests for Motion-X++ → SMPL-X pose mapping (mesh preview helpers)."""

import numpy as np

from src.smplx_mesh_preview import motion_row_to_smplx_pose_arrays
from src.smplx_to_opensim import SMPLX_MOTION_DIM, SMPLX_SLICES


def test_motion_row_to_smplx_pose_arrays_shapes() -> None:
    fe = SMPLX_SLICES["face_expr"]
    row = np.zeros(SMPLX_MOTION_DIM, dtype=np.float32)
    row[fe] = np.linspace(0, 1, fe.stop - fe.start, dtype=np.float32)
    out = motion_row_to_smplx_pose_arrays(row, num_expression_coeffs=50)
    assert out["global_orient"].shape == (3,)
    assert out["body_pose"].shape == (63,)
    assert out["left_hand_pose"].shape == (45,)
    assert out["right_hand_pose"].shape == (45,)
    assert out["jaw_pose"].shape == (3,)
    assert out["expression"].shape == (50,)
    assert out["transl"].shape == (3,)
    assert out["betas"].shape == (10,)


def test_motion_row_expression_padding() -> None:
    fe = SMPLX_SLICES["face_expr"]
    row = np.zeros(SMPLX_MOTION_DIM, dtype=np.float32)
    row[fe.start : fe.start + 10] = 0.25
    out = motion_row_to_smplx_pose_arrays(row, num_expression_coeffs=20)
    assert out["expression"].shape == (20,)
    assert np.allclose(out["expression"][:10], 0.25)
    assert np.allclose(out["expression"][10:], 0.0)


def test_camera_pose_identity_up() -> None:
    from src.smplx_mesh_preview import _camera_pose_world_from_camera_gl

    eye = np.array([2.0, 1.0, 2.0], dtype=np.float64)
    center = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    pose = _camera_pose_world_from_camera_gl(eye, center, np.array([0.0, 1.0, 0.0]))
    assert pose.shape == (4, 4)
    assert np.allclose(pose[3, :], [0.0, 0.0, 0.0, 1.0])
    r = pose[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-5)
    assert np.isclose(float(np.linalg.det(r)), 1.0, rtol=1e-5, atol=1e-5)
