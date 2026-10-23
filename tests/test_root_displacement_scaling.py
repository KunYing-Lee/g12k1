from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from kinematics import load_root_displacement_scale, scale_keypoint_frame_displacements
from solver import project_root_trajectory_axes


def test_preserve_horizontal_and_scale_vertical_root_trajectory(tmp_path: Path) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        "root_displacement_scaling:\n"
        "  x: preserve_source\n"
        "  y: preserve_source\n"
        "  z: leg_length\n",
        encoding="utf-8",
    )
    scales, modes = load_root_displacement_scale(config_path, 0.65)
    np.testing.assert_allclose(scales, [1.0, 1.0, 0.65])
    assert modes == ("preserve_source", "preserve_source", "leg_length")

    keypoints = np.asarray(
        [
            [[1.0, 2.0, 0.5], [2.0, 2.0, 0.5]],
            [[2.0, 4.0, 1.5], [3.0, 4.0, 1.5]],
            [[4.0, 3.0, 3.5], [5.0, 3.0, 3.5]],
        ],
        dtype=np.float32,
    )
    scaled = scale_keypoint_frame_displacements(keypoints, scales)

    np.testing.assert_allclose(scaled[:, 0, :2], keypoints[:, 0, :2])
    np.testing.assert_allclose(scaled[:, 0, 2], [0.5, 1.15, 2.45])
    np.testing.assert_allclose(scaled[:, 1] - scaled[:, 0], keypoints[:, 1] - keypoints[:, 0])


def test_missing_config_preserves_historical_uniform_scaling(tmp_path: Path) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot_xml_path: robot.xml\n", encoding="utf-8")
    scales, modes = load_root_displacement_scale(config_path, 0.7)
    np.testing.assert_allclose(scales, [0.7, 0.7, 0.7])
    assert modes == ("leg_length", "leg_length", "leg_length")


def test_invalid_axis_mode_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        "root_displacement_scaling:\n  x: normalize_speed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="preserve_source"):
        load_root_displacement_scale(config_path, 0.7)


def test_root_trajectory_projection_is_exact_on_selected_axes() -> None:
    qpos = np.asarray([9.0, 8.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.3])
    projected = project_root_trajectory_axes(
        qpos,
        source_root_position=np.asarray([1.25, -2.5, 4.0]),
        free_qposadr=0,
        axes=["x", "y"],
    )
    np.testing.assert_array_equal(projected[:3], [1.25, -2.5, 7.0])
    np.testing.assert_array_equal(qpos[:3], [9.0, 8.0, 7.0])
