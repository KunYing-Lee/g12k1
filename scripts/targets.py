"""Generate the unchanged R9 Cartesian/contact targets from G1 CSV."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import mujoco
import yaml
from scipy.spatial.transform import Rotation
from kinematics import (
    REPLAY_BODY_NAMES,
    apply_link_scales_to_positions,
    average_quaternions_wxyz,
    compute_contact_height_offsets,
    compute_leg_displacement_scale,
    compute_link_geometry_from_positions,
    compute_link_scale_factors,
    compute_robot_body_local_offset,
    compute_robot_link_lengths,
    compute_windowed_point_speeds,
    derive_leg_body_chain_from_links,
    iter_progress,
    load_key_frame_config,
    load_path_config,
    load_root_displacement_scale,
    load_robot_links_config,
    load_scalar_float_config,
    load_scalar_int_config,
    quat_rotate_vectors_wxyz,
    save_keypoints_pkl,
    scale_keypoint_frame_displacements,
    select_frame_slice,
    slice_frame_aligned_arrays,
)

SOURCE_SEMANTIC_LINKS = {
	"left_hip": ("hips_mean", "left_up_leg"),
	"left_thigh": ("left_up_leg", "left_leg"),
	"left_calf": ("left_leg", "left_foot"),
	"right_hip": ("hips_mean", "right_up_leg"),
	"right_thigh": ("right_up_leg", "right_leg"),
	"right_calf": ("right_leg", "right_foot"),
	"neck": ("hips_mean", "shoulder_mean"),
	"head": ("shoulder_mean", "head"),
	"left_shoulder": ("shoulder_mean", "left_arm"),
	"left_arm": ("left_arm", "left_fore_arm"),
	"left_fore_arm": ("left_fore_arm", "left_hand"),
	"right_shoulder": ("shoulder_mean", "right_arm"),
	"right_arm": ("right_arm", "right_fore_arm"),
	"right_fore_arm": ("right_fore_arm", "right_hand"),
}
DEFAULT_SOURCE_ROBOT_CONFIG_PATH = Path("config/robot/g1.yaml")
DEFAULT_TARGET_ROBOT_CONFIG_PATH = Path("config/robot/booster_k1.yaml")
DEFAULT_MOTION_FILE = Path("output_data/robot_motion/bones_g1/body_check_001__A548_M.csv")
DEFAULT_OUTPUT_DIR = Path("output_data/keypoints")
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

def resolve_motion_file(motion_file: Path | None, source_robot_config_path: Path) -> Path:
	if motion_file is not None:
		resolved = motion_file.expanduser().resolve()
		if not resolved.exists():
			raise FileNotFoundError(f"Robot motion CSV not found: {resolved}")
		return resolved

	keypoints_path = load_path_config(source_robot_config_path, "keypoints_path").expanduser().resolve()
	keypoint_stem = keypoints_path.stem
	if keypoint_stem.endswith("_keypoints"):
		keypoint_stem = keypoint_stem[: -len("_keypoints")]
	derived = Path("output_data/robot_motion") / f"{keypoint_stem}_{source_robot_config_path.stem}.csv"
	resolved = derived.expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(
			"Robot motion CSV not found. Provide --motion-file or generate it first with solver.py: "
			f"{resolved}"
		)
	return resolved

def load_robot_motion_csv(csv_path: Path) -> np.ndarray:
	data = np.loadtxt(csv_path, delimiter=",")
	if data.ndim == 1:
		data = data[None, :]
	if data.shape[1] < 7:
		raise ValueError(f"Motion CSV must contain at least 7 columns, got {data.shape}")

	qpos = data.astype(np.float32, copy=True)
	quat_xyzw = qpos[:, 3:7].copy()
	qpos[:, 3] = quat_xyzw[:, 3]
	qpos[:, 4] = quat_xyzw[:, 0]
	qpos[:, 5] = quat_xyzw[:, 1]
	qpos[:, 6] = quat_xyzw[:, 2]
	return qpos

def load_robot_motion_fps(source_robot_config_path: Path, fps_override: float) -> float:
	if fps_override > 0.0:
		return float(fps_override)

	keypoints_path = load_path_config(source_robot_config_path, "keypoints_path").expanduser().resolve()
	if keypoints_path.exists():
		with keypoints_path.open("rb") as f:
			payload = pickle.load(f)
		fps = float(payload.get("fps", 0.0))
		if fps > 0.0:
			return fps
	return 30.0

def load_yaml_body_list_config(config_path: Path, field_name: str) -> tuple[str, ...]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with config_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	value = config.get(field_name)
	if not isinstance(value, list) or not value:
		raise ValueError(f"Missing or invalid list field '{field_name}' in: {config_path}")
	items = tuple(str(item).strip() for item in value if str(item).strip())
	if not items:
		raise ValueError(f"'{field_name}' must contain at least one body name in: {config_path}")
	return items

def load_scalar_bool_config(config_path: Path, field_name: str, default: bool = False) -> bool:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with config_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	value = config.get(field_name, default)
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		parsed = value.strip().lower()
		if parsed in {"true", "1", "yes", "on"}:
			return True
		if parsed in {"false", "0", "no", "off"}:
			return False
	raise ValueError(f"Missing or invalid bool field '{field_name}' in: {config_path}")

def load_effective_knee_angle_offset_degrees(config_path: Path) -> float:
	enabled = load_scalar_bool_config(
		config_path,
		"enable_knee_angle_offset_degrees",
		default=False,
	)
	if not enabled:
		return 0.0
	return load_scalar_float_config(config_path, "knee_angle_offset_degrees", default=0.0)

def validate_motion_shape(model: mujoco.MjModel, qpos_frames: np.ndarray, motion_file: Path) -> None:
	if qpos_frames.ndim != 2:
		raise ValueError(f"Motion CSV must be a 2D array, got {qpos_frames.shape}")
	if qpos_frames.shape[1] != model.nq:
		raise ValueError(
			f"Motion CSV qpos dim mismatch for {motion_file}: expected {model.nq}, got {qpos_frames.shape[1]}"
		)

def canonicalize_contact_name(body_name: str) -> str:
	lower = body_name.lower()
	if "left" in lower and "toe" in lower:
		return "left_toe"
	if "right" in lower and "toe" in lower:
		return "right_toe"
	if "left" in lower and "foot" in lower and "end" in lower:
		return "left_foot_end"
	if "right" in lower and "foot" in lower and "end" in lower:
		return "right_foot_end"
	if "left" in lower and ("hand" in lower or "wrist" in lower):
		return "left_hand"
	if "right" in lower and ("hand" in lower or "wrist" in lower):
		return "right_hand"
	return body_name

def resolve_source_semantic_body_mapping(
	source_robot_links: dict[str, tuple[str, str]],
) -> dict[str, str | tuple[str, str]]:
	required_links = (
		"left_hip",
		"left_thigh",
		"left_calf",
		"right_hip",
		"right_thigh",
		"right_calf",
		"neck",
		"head",
		"left_shoulder",
		"left_arm",
		"left_fore_arm",
		"right_shoulder",
		"right_arm",
		"right_fore_arm",
	)
	for link_name in required_links:
		if link_name not in source_robot_links:
			raise ValueError(f"Missing source robot link required for replay: {link_name}")

	return {
		"hips_mean": (source_robot_links["left_hip"][0], source_robot_links["right_hip"][0]),
		"hips": source_robot_links["left_hip"][0],
		"left_up_leg": source_robot_links["left_hip"][1],
		"left_leg": source_robot_links["left_thigh"][1],
		"left_foot": source_robot_links["left_calf"][1],
		"right_up_leg": source_robot_links["right_hip"][1],
		"right_leg": source_robot_links["right_thigh"][1],
		"right_foot": source_robot_links["right_calf"][1],
		"shoulder_mean": (
			source_robot_links["left_shoulder"][0],
			source_robot_links["right_shoulder"][0],
		),
		"spine1": source_robot_links["neck"][0],
		"spine2": source_robot_links["neck"][0],
		"chest": source_robot_links["neck"][1],
		"neck": source_robot_links["neck"][1],
		"head": source_robot_links["head"][1],
		"left_shoulder": source_robot_links["left_shoulder"][1],
		"left_arm": source_robot_links["left_shoulder"][1],
		"left_fore_arm": source_robot_links["left_arm"][1],
		"left_hand": source_robot_links["left_fore_arm"][1],
		"right_shoulder": source_robot_links["right_shoulder"][1],
		"right_arm": source_robot_links["right_shoulder"][1],
		"right_fore_arm": source_robot_links["right_arm"][1],
		"right_hand": source_robot_links["right_fore_arm"][1],
	}

def get_body_id(model: mujoco.MjModel, body_name: str) -> int:
	body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
	if body_id < 0:
		raise ValueError(f"Missing body in MJCF: {body_name}")
	return int(body_id)

def build_targets_buffers(
	qpos_frames: np.ndarray,
	source_robot_mjcf_path: Path,
	source_robot_links: dict[str, tuple[str, str]],
	contact_body_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	model = mujoco.MjModel.from_xml_path(str(source_robot_mjcf_path))
	data = mujoco.MjData(model)
	validate_motion_shape(model, qpos_frames, source_robot_mjcf_path)

	body_slots = {name: idx for idx, name in enumerate(REPLAY_BODY_NAMES)}
	semantic_mapping = resolve_source_semantic_body_mapping(source_robot_links)
	resolved_body_ids: dict[str, int | tuple[int, int]] = {}
	for semantic_name, body_spec in semantic_mapping.items():
		if isinstance(body_spec, tuple):
			resolved_body_ids[semantic_name] = tuple(get_body_id(model, name) for name in body_spec)
		else:
			resolved_body_ids[semantic_name] = get_body_id(model, body_spec)

	contact_body_ids = [get_body_id(model, body_name) for body_name in contact_body_names]
	positions = np.zeros((qpos_frames.shape[0], len(REPLAY_BODY_NAMES), 3), dtype=np.float32)
	quaternions = np.broadcast_to(IDENTITY_QUAT_WXYZ, (qpos_frames.shape[0], len(REPLAY_BODY_NAMES), 4)).copy()
	contact_positions = np.zeros((qpos_frames.shape[0], len(contact_body_ids), 3), dtype=np.float32)

	frame_iter = iter_progress(
		enumerate(qpos_frames),
		total=qpos_frames.shape[0],
		desc="Robot body replay",
		unit="frame",
	)
	for frame_idx, qpos in frame_iter:
		data.qpos[:] = qpos
		mujoco.mj_forward(model, data)

		for semantic_name, slot in body_slots.items():
			if semantic_name not in resolved_body_ids:
				continue
			body_spec = resolved_body_ids[semantic_name]
			if isinstance(body_spec, tuple):
				left_id, right_id = body_spec
				positions[frame_idx, slot, :] = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
				quaternions[frame_idx, slot, :] = average_quaternions_wxyz(
					data.xquat[left_id][None, :].astype(np.float32),
					data.xquat[right_id][None, :].astype(np.float32),
				)[0]
			else:
				positions[frame_idx, slot, :] = data.xpos[body_spec]
				quaternions[frame_idx, slot, :] = data.xquat[body_spec]

		for contact_idx, body_id in enumerate(contact_body_ids):
			contact_positions[frame_idx, contact_idx, :] = data.xpos[body_id]

	return positions.astype(np.float32), quaternions.astype(np.float32), contact_positions

def compute_robot_contact_sequence(
	contact_positions: np.ndarray,
	fps: float,
	vel_window: int,
	vel_threshold: float,
	height_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
	contact_speeds = compute_windowed_point_speeds(
		contact_positions,
		fps=fps,
		window=vel_window,
		progress_desc="Contact speed",
	)
	contact_states = np.logical_and(
		contact_speeds <= float(vel_threshold),
		contact_positions[:, :, 2] <= float(height_threshold),
	)
	return contact_speeds.astype(np.float32), contact_states.astype(np.bool_)

def append_target_contact_keypoints(
	keypoints: np.ndarray,
	keypoint_quaternions: np.ndarray,
	anchor_raw_quaternions: np.ndarray,
	target_robot_links: dict[str, tuple[str, str]],
	target_robot_config_path: Path,
	target_robot_mjcf_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
	target_contact_links = load_yaml_body_list_config(target_robot_config_path, "contact_links")
	contact_name_to_body = {
		canonicalize_contact_name(body_name): body_name for body_name in target_contact_links
	}
	anchor_specs = (
		("left_foot_end", "left_calf"),
		("right_foot_end", "right_calf"),
		("left_toe", "left_calf"),
		("right_toe", "right_calf"),
	)
	keypoint_index = {name: idx for idx, name in enumerate(["hips_mean", *target_robot_links.keys()])}
	extra_positions: list[np.ndarray] = []
	extra_quaternions: list[np.ndarray] = []
	extra_names: list[str] = []

	for contact_name, anchor_link_name in anchor_specs:
		target_body_name = contact_name_to_body.get(contact_name)
		if target_body_name is None:
			continue
		if anchor_link_name not in target_robot_links:
			raise ValueError(f"Missing target robot link required for extra contact keypoints: {anchor_link_name}")
		anchor_body_name = target_robot_links[anchor_link_name][1]
		anchor_idx = keypoint_index[anchor_link_name]
		anchor_positions = keypoints[:, anchor_idx, :]
		anchor_quaternions = keypoint_quaternions[:, anchor_idx, :]
		anchor_raw_quat = anchor_raw_quaternions[:, anchor_idx, :]
		local_offset = compute_robot_body_local_offset(
			target_robot_mjcf_path,
			anchor_body=anchor_body_name,
			target_body=target_body_name,
		)
		world_offset = quat_rotate_vectors_wxyz(anchor_raw_quat, local_offset)
		extra_positions.append(anchor_positions + world_offset)
		extra_quaternions.append(anchor_quaternions)
		extra_names.append(target_body_name)

	if not extra_positions:
		return keypoints, keypoint_quaternions, extra_names

	extra_positions_arr = np.stack(extra_positions, axis=1)
	extra_quaternions_arr = np.stack(extra_quaternions, axis=1)
	return (
		np.concatenate([keypoints, extra_positions_arr], axis=1),
		np.concatenate([keypoint_quaternions, extra_quaternions_arr], axis=1),
		extra_names,
	)

def build_key_frame_adjustment_matrix(
	axis_map: np.ndarray,
	euler_xyz_degrees: np.ndarray,
) -> np.ndarray:
	if axis_map.shape != (3, 3):
		raise ValueError(f"axis_map shape must be (3, 3), got {axis_map.shape}")
	axis_map64 = axis_map.astype(np.float64)
	euler_rad = np.radians(euler_xyz_degrees.astype(np.float64))
	rx, ry, rz = float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2])
	rot_x = np.array(
		[[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]],
		dtype=np.float64,
	)
	rot_y = np.array(
		[[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]],
		dtype=np.float64,
	)
	rot_z = np.array(
		[[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]],
		dtype=np.float64,
	)
	return axis_map64 @ rot_x @ rot_y @ rot_z

def transform_key_frame_quaternions_wxyz(
	quaternions_wxyz: np.ndarray,
	source_axis_map: np.ndarray,
	source_euler_xyz_degrees: np.ndarray,
	target_axis_map: np.ndarray,
	target_euler_xyz_degrees: np.ndarray,
) -> np.ndarray:
	base_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
	base_mats = Rotation.from_quat(base_xyzw).as_matrix()
	source_adjust = build_key_frame_adjustment_matrix(source_axis_map, source_euler_xyz_degrees)
	target_adjust = build_key_frame_adjustment_matrix(target_axis_map, target_euler_xyz_degrees)
	adjusted_mats = np.einsum(
		"nij,jk,kl->nil",
		base_mats,
		np.linalg.inv(source_adjust),
		target_adjust,
	)
	adjusted_xyzw = Rotation.from_matrix(adjusted_mats).as_quat()
	return adjusted_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

def derive_rest_frame_calibrations(source_config_path, target_config_path, target_mjcf):
	"""Map anatomical link directions, not coincident MJCF frame labels.

	Each frame uses the outgoing segment as its longitudinal axis; terminal
	frames use the incoming segment. The robot root forward axis fixes twist.
	The resulting constant right rotation maps target-local axes into source
	local axes and is independent of the motion being retargeted.
	"""
	with source_config_path.open() as f: source_config = yaml.safe_load(f)
	with target_config_path.open() as f: target_config = yaml.safe_load(f)
	selected = target_config.get("rest_frame_calibration_links", [])
	if not selected: return {}
	source_path = Path(source_config["robot_xml_path"])
	if not source_path.is_absolute(): source_path = Path(__file__).resolve().parents[1] / source_path
	models = [mujoco.MjModel.from_xml_path(str(source_path)), mujoco.MjModel.from_xml_path(str(target_mjcf))]
	bases = []
	for model, config in zip(models, [source_config, target_config]):
		data = mujoco.MjData(model); mujoco.mj_forward(model, data)
		links = config["robot_links"]; frames = {}
		root_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
		root = model.jnt_bodyid[root_joint]; forward = data.xmat[root].reshape(3,3)[:,0]
		for name in selected:
			parent, body = links[name]; bid = model.body(body).id
			outgoing = [pair[1] for pair in links.values() if pair[0] == body]
			if len(outgoing) > 1: raise ValueError(f"Ambiguous anatomical axis for {name}")
			vector = (data.xpos[model.body(outgoing[0]).id] - data.xpos[bid] if outgoing
				else data.xpos[bid] - data.xpos[model.body(parent).id])
			axis = vector / np.linalg.norm(vector)
			secondary = forward - axis * (axis @ forward)
			if np.linalg.norm(secondary) < 1e-8: raise ValueError(f"Degenerate anatomical frame for {name}")
			secondary /= np.linalg.norm(secondary)
			frames[name] = data.xmat[bid].reshape(3,3).T @ np.column_stack((axis, secondary, np.cross(axis, secondary)))
		bases.append(frames)
	return {name: bases[0][name] @ bases[1][name].T for name in selected}

def build_solver_keypoints(
	source_positions: np.ndarray,
	source_quaternions: np.ndarray,
	target_robot_link_lengths: dict[str, float],
	source_robot_config_path: Path,
	target_robot_config_path: Path,
	target_robot_mjcf_path: Path,
) -> tuple[
		np.ndarray,
		np.ndarray,
		np.ndarray,
		dict[str, np.ndarray],
		dict[str, float | np.ndarray],
		dict[str, bool],
	]:
	body_slots = {name: idx for idx, name in enumerate(REPLAY_BODY_NAMES)}
	calibrations = derive_rest_frame_calibrations(source_robot_config_path, target_robot_config_path, target_robot_mjcf_path)
	target_robot_links = load_robot_links_config(target_robot_config_path)
	source_key_frame_offset_degrees, source_key_frame_axis_map = load_key_frame_config(
		source_robot_config_path,
		section_name="key_frame_config",
	)
	knee_angle_offset_degrees = load_effective_knee_angle_offset_degrees(target_robot_config_path)
	key_frame_offset_degrees, key_frame_axis_map = load_key_frame_config(
		target_robot_config_path,
		section_name="key_frame_config",
	)
	left_leg_links = derive_leg_body_chain_from_links(SOURCE_SEMANTIC_LINKS, "left")
	right_leg_links = derive_leg_body_chain_from_links(SOURCE_SEMANTIC_LINKS, "right")
	source_link_lengths, source_link_vectors, resolved_source_links = compute_link_geometry_from_positions(
		SOURCE_SEMANTIC_LINKS,
		body_slots,
		source_positions,
	)
	link_scales, link_scale_is_static = compute_link_scale_factors(
		target_robot_link_lengths,
		source_link_lengths,
	)
	retarget_keypoints, retarget_quaternions = apply_link_scales_to_positions(
		source_positions,
		resolved_source_links,
		body_slots,
		link_scales,
		source_link_vectors,
		knee_angle_offset_degrees,
		left_leg_links,
		right_leg_links,
		quaternions=source_quaternions,
	)
	ordered_keypoints = np.zeros((source_positions.shape[0], len(target_robot_links) + 1, 3), dtype=np.float32)
	ordered_quaternions = np.zeros((source_positions.shape[0], len(target_robot_links) + 1, 4), dtype=np.float32)
	ordered_raw_quaternions = np.zeros((source_positions.shape[0], len(target_robot_links) + 1, 4), dtype=np.float32)
	ordered_keypoints[:, 0, :] = retarget_keypoints[:, body_slots["hips_mean"], :]
	source_hips_mean_offset = source_key_frame_offset_degrees.get("hips_mean", np.zeros(3, dtype=np.float32))
	source_hips_mean_axis_map = source_key_frame_axis_map.get("hips_mean", np.eye(3, dtype=np.float32))
	hips_mean_offset = key_frame_offset_degrees.get("hips_mean", np.zeros(3, dtype=np.float32))
	hips_mean_axis_map = key_frame_axis_map.get("hips_mean", np.eye(3, dtype=np.float32))
	ordered_raw_quaternions[:, 0, :] = transform_key_frame_quaternions_wxyz(
		retarget_quaternions[:, body_slots["hips_mean"], :],
		source_hips_mean_axis_map,
		source_hips_mean_offset,
		hips_mean_axis_map,
		hips_mean_offset,
	)
	ordered_quaternions[:, 0, :] = ordered_raw_quaternions[:, 0, :]
	for keypoint_idx, link_name in enumerate(target_robot_links, start=1):
		if link_name not in resolved_source_links:
			raise ValueError(f"Missing source semantic link for target robot link: {link_name}")
		_parent_body, child_body = resolved_source_links[link_name]
		child_idx = body_slots[child_body]
		ordered_keypoints[:, keypoint_idx, :] = retarget_keypoints[:, child_idx, :]
		source_euler_offset = source_key_frame_offset_degrees.get(child_body, np.zeros(3, dtype=np.float32))
		source_axis_map = source_key_frame_axis_map.get(child_body, np.eye(3, dtype=np.float32))
		euler_offset = key_frame_offset_degrees.get(child_body, np.zeros(3, dtype=np.float32))
		axis_map = key_frame_axis_map.get(child_body, np.eye(3, dtype=np.float32))
		ordered_raw_quaternions[:, keypoint_idx, :] = transform_key_frame_quaternions_wxyz(
			retarget_quaternions[:, child_idx, :],
			source_axis_map,
			source_euler_offset,
			axis_map,
			euler_offset,
		)
		if link_name in calibrations:
			raw = Rotation.from_quat(retarget_quaternions[:, child_idx, :][:, [1,2,3,0]]).as_matrix()
			quat = Rotation.from_matrix(raw @ calibrations[link_name]).as_quat()
			ordered_raw_quaternions[:, keypoint_idx, :] = quat[:, [3,0,1,2]]
		ordered_quaternions[:, keypoint_idx, :] = ordered_raw_quaternions[:, keypoint_idx, :]

	with target_robot_config_path.open("r", encoding="utf-8") as f:
		target_config = yaml.safe_load(f) or {}
	raw_target_rest_anchor_links = target_config.get("target_rest_anchor_links", [])
	if not isinstance(raw_target_rest_anchor_links, list):
		raise ValueError(
			"target_rest_anchor_links must be a list of robot_links names in "
			f"{target_robot_config_path}"
		)
	target_rest_anchor_links = tuple(
		str(link_name).strip()
		for link_name in raw_target_rest_anchor_links
		if str(link_name).strip()
	)
	if target_rest_anchor_links:
		link_indices = {link_name: idx for idx, link_name in enumerate(target_robot_links, start=1)}
		semantic_child_to_link = {
			child_body: link_name
			for link_name, (_parent_body, child_body) in resolved_source_links.items()
		}

		def collect_descendant_links(root_link_name: str) -> tuple[str, ...]:
			root_child = resolved_source_links[root_link_name][1]
			descendants = [root_link_name]
			frontier = [root_child]
			while frontier:
				parent_semantic = frontier.pop()
				for candidate_name, (candidate_parent, candidate_child) in resolved_source_links.items():
					if candidate_parent != parent_semantic:
						continue
					descendants.append(candidate_name)
					frontier.append(candidate_child)
			return tuple(descendants)

		for link_name in target_rest_anchor_links:
			if link_name not in target_robot_links:
				raise ValueError(f"Unknown target_rest_anchor_links entry: {link_name}")
			parent_semantic = resolved_source_links[link_name][0]
			if parent_semantic == "hips_mean":
				parent_idx = 0
			else:
				parent_link_name = semantic_child_to_link.get(parent_semantic)
				if parent_link_name is None:
					raise ValueError(
						f"Could not resolve semantic parent '{parent_semantic}' for anchor link '{link_name}'"
					)
				parent_idx = link_indices[parent_link_name]

			child_idx = link_indices[link_name]
			start_body, end_body = target_robot_links[link_name]
			local_rest_offset = compute_robot_body_local_offset(
				target_robot_mjcf_path,
				anchor_body=start_body,
				target_body=end_body,
			)
			world_rest_offset = quat_rotate_vectors_wxyz(
				ordered_raw_quaternions[:, parent_idx, :],
				local_rest_offset,
			)
			desired_child_positions = ordered_keypoints[:, parent_idx, :] + world_rest_offset
			translation = desired_child_positions - ordered_keypoints[:, child_idx, :]
			for descendant_name in collect_descendant_links(link_name):
				ordered_keypoints[:, link_indices[descendant_name], :] += translation

	return (
		ordered_keypoints,
		ordered_quaternions,
		ordered_raw_quaternions,
		source_link_vectors,
		link_scales,
		link_scale_is_static,
	)

def offset_keypoints_by_contact_height(
	keypoints: np.ndarray,
	keypoint_names: list[str],
	contact_names: tuple[str, ...],
	contact_positions: np.ndarray,
	contact_states: np.ndarray,
	height_lpf_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
	height_offsets = compute_contact_height_offsets(
		keypoints=keypoints,
		keypoint_names=keypoint_names,
		contact_names=contact_names,
		contact_positions=contact_positions,
		contact_states=contact_states,
		height_lpf_alpha=height_lpf_alpha,
	)
	adjusted_keypoints = keypoints.copy()
	adjusted_keypoints[:, :, 2] -= height_offsets[:, None]
	return adjusted_keypoints.astype(np.float32), height_offsets.astype(np.float32)

def build_output_path(
	output_path: Path | None,
	motion_file: Path,
	source_robot_config_path: Path,
	target_robot_config_path: Path,
) -> Path:
	if output_path is not None:
		return output_path.expanduser().resolve()
	return (
		(DEFAULT_OUTPUT_DIR / target_robot_config_path.stem / f"{motion_file.stem}_from_{source_robot_config_path.stem}_keypoints.pkl")
		.expanduser()
		.resolve()
	)

def prepare_keypoints(motion_file: Path, output_path: Path) -> None:
	from types import SimpleNamespace
	args = SimpleNamespace(source_robot_config=Path("config/robot/g1.yaml"),
	    target_robot_config=Path("config/robot/booster_k1.yaml"), motion_file=motion_file,
	    output_path=output_path, fps=120., start_frame=0, end_frame=-1, stride=1)
	source_robot_config_path = args.source_robot_config.expanduser().resolve()
	target_robot_config_path = args.target_robot_config.expanduser().resolve()
	motion_file = resolve_motion_file(args.motion_file, source_robot_config_path)

	source_robot_mjcf_path = load_path_config(source_robot_config_path, "robot_xml_path").expanduser().resolve()
	target_robot_mjcf_path = load_path_config(target_robot_config_path, "robot_xml_path").expanduser().resolve()
	source_robot_links = load_robot_links_config(source_robot_config_path)
	target_robot_links = load_robot_links_config(target_robot_config_path)
	source_contact_links = load_yaml_body_list_config(source_robot_config_path, "contact_links")
	contact_names = tuple(canonicalize_contact_name(name) for name in source_contact_links)

	qpos_frames = load_robot_motion_csv(motion_file)
	positions, quaternions, contact_positions = build_targets_buffers(
		qpos_frames=qpos_frames,
		source_robot_mjcf_path=source_robot_mjcf_path,
		source_robot_links=source_robot_links,
		contact_body_names=source_contact_links,
	)
	fps = load_robot_motion_fps(source_robot_config_path, args.fps)
	contact_vel_window = load_scalar_int_config(
		source_robot_config_path,
		"contact_vel_calculate_window",
		default=6,
	)
	contact_vel_threshold = load_scalar_float_config(
		source_robot_config_path,
		"contact_vel_threshold",
		default=0.5,
	)
	contact_height_threshold = load_scalar_float_config(
		source_robot_config_path,
		"contact_height_threshold",
		default=0.025,
	)
	contact_height_lpf_alpha = load_scalar_float_config(
		source_robot_config_path,
		"contact_height_lpf_alpha",
		default=0.2,
	)
	contact_speeds, contact_states = compute_robot_contact_sequence(
		contact_positions=contact_positions,
		fps=fps,
		vel_window=contact_vel_window,
		vel_threshold=contact_vel_threshold,
		height_threshold=contact_height_threshold,
	)

	target_robot_link_lengths = compute_robot_link_lengths(
		config_path=target_robot_config_path,
		robot_mjcf_path=target_robot_mjcf_path,
	)
	(
		retarget_keypoints,
		retarget_keypoint_quaternions,
		retarget_raw_quaternions,
		source_link_vectors,
		link_scales,
		link_scale_is_static,
	) = build_solver_keypoints(
		source_positions=positions,
		source_quaternions=quaternions,
		target_robot_link_lengths=target_robot_link_lengths,
		source_robot_config_path=source_robot_config_path,
		target_robot_config_path=target_robot_config_path,
		target_robot_mjcf_path=target_robot_mjcf_path,
	)
	knee_angle_offset_degrees = load_effective_knee_angle_offset_degrees(target_robot_config_path)
	(
		retarget_keypoints,
		retarget_keypoint_quaternions,
		extra_keypoint_names,
	) = append_target_contact_keypoints(
		keypoints=retarget_keypoints,
		keypoint_quaternions=retarget_keypoint_quaternions,
		anchor_raw_quaternions=retarget_raw_quaternions,
		target_robot_links=target_robot_links,
		target_robot_config_path=target_robot_config_path,
		target_robot_mjcf_path=target_robot_mjcf_path,
	)
	leg_displacement_scale, robot_leg_length, source_leg_length = compute_leg_displacement_scale(
		robot_link_lengths=target_robot_link_lengths,
		skeleton_link_vectors=source_link_vectors,
		knee_angle_offset_degrees=knee_angle_offset_degrees,
	)
	root_displacement_scale, root_displacement_scaling = load_root_displacement_scale(
		target_robot_config_path,
		leg_displacement_scale,
	)
	retarget_keypoints = scale_keypoint_frame_displacements(
		keypoints=retarget_keypoints,
		displacement_scale=root_displacement_scale,
		root_keypoint_idx=0,
	)
	keypoint_names = ["hips_mean", *list(target_robot_links.keys()), *extra_keypoint_names]
	retarget_keypoints, contact_height_offsets = offset_keypoints_by_contact_height(
		keypoints=retarget_keypoints,
		keypoint_names=keypoint_names,
		contact_names=contact_names,
		contact_positions=contact_positions,
		contact_states=contact_states,
		height_lpf_alpha=contact_height_lpf_alpha,
	)
	frame_ids = select_frame_slice(positions.shape[0], args.start_frame, args.end_frame, args.stride)
	export_fps = fps / args.stride
	selected = slice_frame_aligned_arrays(
		frame_ids,
		positions=retarget_keypoints,
		quaternions=retarget_keypoint_quaternions,
		contact_positions=contact_positions,
		contact_speeds=contact_speeds,
		contact_states=contact_states,
		source_root_positions=qpos_frames[:, :3],
	)
	keypoint_output_path = build_output_path(
		output_path=args.output_path,
		motion_file=motion_file,
		source_robot_config_path=source_robot_config_path,
		target_robot_config_path=target_robot_config_path,
	)
	save_keypoints_pkl(
		output_path=keypoint_output_path,
		keypoint_names=keypoint_names,
		positions=selected["positions"],
		quaternions=selected["quaternions"],
		fps=export_fps,
		contact_names=contact_names,
		contact_positions=selected["contact_positions"],
		contact_speeds=selected["contact_speeds"],
		contact_states=selected["contact_states"],
		source_root_positions=selected["source_root_positions"],
		contact_vel_window=contact_vel_window,
		contact_vel_threshold=contact_vel_threshold,
		contact_height_threshold=contact_height_threshold,
	)

	# Preserve the observed source articulation for anatomical pose objectives.
	# CSV frame selection is shared exactly with Cartesian targets.
	source_model = mujoco.MjModel.from_xml_path(str(source_robot_mjcf_path))
	joint_ids = [j for j in range(source_model.njnt) if source_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
	with keypoint_output_path.open('rb') as f: payload = pickle.load(f)
	payload['source_joint_names'] = [source_model.joint(j).name for j in joint_ids]
	payload['source_joint_positions'] = qpos_frames[frame_ids][:, source_model.jnt_qposadr[joint_ids]].copy()
	with keypoint_output_path.open('wb') as f: pickle.dump(payload, f)
