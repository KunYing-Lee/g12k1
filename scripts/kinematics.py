"""Kinematic transforms and target serialization used by the fixed G1/K1 pipeline."""
from __future__ import annotations
import pickle
from pathlib import Path
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

REPLAY_BODY_NAMES = ['hips', 'left_up_leg', 'left_leg', 'left_foot', 'left_toe', 'right_up_leg', 'right_leg', 'right_foot', 'right_toe', 'spine1', 'spine2', 'chest', 'neck', 'head', 'left_shoulder', 'left_arm', 'left_fore_arm', 'left_hand', 'right_shoulder', 'right_arm', 'right_fore_arm', 'right_hand', 'hips_mean', 'shoulder_mean']

def iter_progress(
	iterable,
	*,
	total: int | None = None,
	desc: str | None = None,
	unit: str | None = None,
):
	return tqdm(
		iterable,
		total=total,
		desc=desc,
		unit=unit,
		dynamic_ncols=True,
		leave=True,
	)

def batch_rotation_between_vectors_wxyz(
	v_from: np.ndarray,
	v_to: np.ndarray,
	eps: float = 1e-8,
) -> np.ndarray:
	"""Compute the minimal world-frame rotation (wxyz quaternions) that maps each
	``v_from`` direction onto the corresponding ``v_to`` direction.

	The rotation axis is the cross product of the two vectors, i.e. the normal of
	the plane spanned by them, so this is exactly a rotation about that plane normal.
	"""
	v_from = np.asarray(v_from, dtype=np.float64)
	v_to = np.asarray(v_to, dtype=np.float64)
	from_norm = np.linalg.norm(v_from, axis=-1, keepdims=True)
	to_norm = np.linalg.norm(v_to, axis=-1, keepdims=True)
	from_u = v_from / np.clip(from_norm, eps, None)
	to_u = v_to / np.clip(to_norm, eps, None)

	axis = np.cross(from_u, to_u)
	axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
	cos_angle = np.clip(np.sum(from_u * to_u, axis=-1, keepdims=True), -1.0, 1.0)
	angle = np.arccos(cos_angle)

	# Pick an arbitrary perpendicular axis for the (near-)antiparallel case so the
	# 180-degree flip stays well defined.
	orthogonal_axis = np.cross(from_u, np.array([1.0, 0.0, 0.0], dtype=np.float64))
	orthogonal_axis_norm = np.linalg.norm(orthogonal_axis, axis=-1, keepdims=True)
	orthogonal_axis = np.where(
		orthogonal_axis_norm > eps,
		orthogonal_axis,
		np.cross(from_u, np.array([0.0, 1.0, 0.0], dtype=np.float64)),
	)
	axis = np.where(axis_norm > eps, axis, orthogonal_axis)
	axis = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), eps, None)

	rotvec = axis * angle
	quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()
	return quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

def multiply_quaternions_wxyz(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
	"""Compose two batches of wxyz quaternions as ``q_left * q_right``."""
	left_xyzw = np.asarray(q_left, dtype=np.float64)[:, [1, 2, 3, 0]]
	right_xyzw = np.asarray(q_right, dtype=np.float64)[:, [1, 2, 3, 0]]
	result_xyzw = (Rotation.from_quat(left_xyzw) * Rotation.from_quat(right_xyzw)).as_quat()
	return result_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

def compute_two_bone_ik_knee_positions(
	hip_positions: np.ndarray,
	knee_positions: np.ndarray,
	foot_positions: np.ndarray,
	target_foot_positions: np.ndarray,
) -> np.ndarray:
	upper_leg = knee_positions - hip_positions
	lower_leg = foot_positions - knee_positions
	upper_len = np.linalg.norm(upper_leg, axis=-1)
	lower_len = np.linalg.norm(lower_leg, axis=-1)

	original_hip_to_foot = foot_positions - hip_positions
	original_hip_to_foot_len = np.linalg.norm(original_hip_to_foot, axis=-1)
	original_u = original_hip_to_foot / np.clip(original_hip_to_foot_len[:, None], 1e-8, None)

	knee_from_hip = knee_positions - hip_positions
	knee_along_u = np.sum(knee_from_hip * original_u, axis=-1, keepdims=True)
	knee_projection = hip_positions + knee_along_u * original_u
	bend_pref = knee_positions - knee_projection
	bend_pref_norm = np.linalg.norm(bend_pref, axis=-1, keepdims=True)
	world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
	bend_pref = np.where(
		bend_pref_norm > 1e-8,
		bend_pref,
		np.cross(original_u, world_up[None, :]),
	)
	bend_pref_norm = np.linalg.norm(bend_pref, axis=-1, keepdims=True)
	bend_pref = np.where(
		bend_pref_norm > 1e-8,
		bend_pref,
		np.cross(original_u, np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]),
	)
	bend_pref = bend_pref / np.clip(np.linalg.norm(bend_pref, axis=-1, keepdims=True), 1e-8, None)

	# Keep blended knee in the original hip-knee-foot plane whenever possible.
	plane_normal = np.cross(upper_leg, original_hip_to_foot)
	plane_normal_norm = np.linalg.norm(plane_normal, axis=-1, keepdims=True)
	plane_normal = np.where(
		plane_normal_norm > 1e-8,
		plane_normal,
		np.cross(original_u, bend_pref),
	)
	plane_normal_norm = np.linalg.norm(plane_normal, axis=-1, keepdims=True)
	plane_normal = np.where(
		plane_normal_norm > 1e-8,
		plane_normal,
		np.cross(original_u, world_up[None, :]),
	)
	plane_normal_norm = np.linalg.norm(plane_normal, axis=-1, keepdims=True)
	plane_normal = np.where(
		plane_normal_norm > 1e-8,
		plane_normal,
		np.cross(original_u, np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]),
	)
	plane_normal = plane_normal / np.clip(np.linalg.norm(plane_normal, axis=-1, keepdims=True), 1e-8, None)

	target_hip_to_foot = target_foot_positions - hip_positions
	target_hip_to_foot_len = np.linalg.norm(target_hip_to_foot, axis=-1)
	target_u = target_hip_to_foot / np.clip(target_hip_to_foot_len[:, None], 1e-8, None)

	d = np.clip(target_hip_to_foot_len, 1e-8, upper_len + lower_len - 1e-8)
	x = (upper_len**2 - lower_len**2 + d**2) / (2.0 * np.clip(d, 1e-8, None))
	h = np.sqrt(np.clip(upper_len**2 - x**2, 0.0, None))

	bend_pref_proj = bend_pref - np.sum(bend_pref * target_u, axis=-1, keepdims=True) * target_u
	bend_dir = np.cross(plane_normal, target_u)
	bend_dir_norm = np.linalg.norm(bend_dir, axis=-1, keepdims=True)
	bend_dir = np.where(
		bend_dir_norm > 1e-8,
		bend_dir,
		bend_pref_proj,
	)
	bend_dir_norm = np.linalg.norm(bend_dir, axis=-1, keepdims=True)
	bend_dir = np.where(
		bend_dir_norm > 1e-8,
		bend_dir,
		np.cross(target_u, np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]),
	)
	bend_dir = bend_dir / np.clip(np.linalg.norm(bend_dir, axis=-1, keepdims=True), 1e-8, None)

	bend_pref_proj_norm = np.linalg.norm(bend_pref_proj, axis=-1, keepdims=True)
	bend_pref_proj = np.where(
		bend_pref_proj_norm > 1e-8,
		bend_pref_proj,
		bend_pref,
	)
	bend_sign = np.where(np.sum(bend_dir * bend_pref_proj, axis=-1) >= 0.0, 1.0, -1.0)
	bend_dir = bend_dir * bend_sign[:, None]

	blended_knee_positions = hip_positions + x[:, None] * target_u + h[:, None] * bend_dir
	return blended_knee_positions.astype(np.float32)

def load_link_pairs_config(config_path: Path, section_name: str) -> dict[str, tuple[str, str]]:
	if not config_path.exists():
		raise FileNotFoundError(f"Link config file not found: {config_path}")

	link_pairs: dict[str, tuple[str, str]] = {}
	in_section = False
	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].rstrip()
		if not line.strip():
			continue

		if not in_section:
			if line.strip() == f"{section_name}:":
				in_section = True
			continue

		if not raw_line[:1].isspace():
			break

		stripped = line.strip()
		link_name, separator, body_spec = stripped.partition(":")
		if not separator:
			raise ValueError(f"Invalid robot_links entry: {raw_line}")

		body_spec = body_spec.strip()
		if not (body_spec.startswith("[") and body_spec.endswith("]")):
			raise ValueError(f"Invalid robot_links body pair: {raw_line}")

		body_names = [item.strip() for item in body_spec[1:-1].split(",")]
		if len(body_names) != 2 or not all(body_names):
			raise ValueError(f"{section_name} entry must contain exactly two body names: {raw_line}")

		link_pairs[link_name.strip()] = (body_names[0], body_names[1])

	if not link_pairs:
		raise ValueError(f"No {section_name} entries found in: {config_path}")
	return link_pairs

def load_robot_links_config(config_path: Path) -> dict[str, tuple[str, str]]:
	return load_link_pairs_config(config_path, "robot_links")

def derive_leg_body_chain_from_links(
	skeleton_links: dict[str, tuple[str, str]],
	leg_side: str,
) -> tuple[str, str, str]:
	thigh_link_name = f"{leg_side}_thigh"
	calf_link_name = f"{leg_side}_calf"
	if thigh_link_name not in skeleton_links:
		raise ValueError(f"Missing skeleton link required for leg chain: {thigh_link_name}")
	if calf_link_name not in skeleton_links:
		raise ValueError(f"Missing skeleton link required for leg chain: {calf_link_name}")

	thigh_parent, thigh_child = skeleton_links[thigh_link_name]
	calf_parent, calf_child = skeleton_links[calf_link_name]
	if thigh_child != calf_parent:
		raise ValueError(
			f"Inconsistent {leg_side} leg links: {thigh_link_name} ends at {thigh_child}, "
			f"but {calf_link_name} starts at {calf_parent}"
		)
	return thigh_parent, thigh_child, calf_child

def load_scalar_float_config(config_path: Path, field_name: str, default: float | None = None) -> float:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if separator and key.strip() == field_name:
			parsed = value.strip()
			if not parsed:
				raise ValueError(f"Missing value for '{field_name}' in: {config_path}")
			return float(parsed)

	if default is not None:
		return float(default)
	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")

def load_scalar_int_config(config_path: Path, field_name: str, default: int | None = None) -> int:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if separator and key.strip() == field_name:
			parsed = value.strip()
			if not parsed:
				raise ValueError(f"Missing value for '{field_name}' in: {config_path}")
			return int(parsed)

	if default is not None:
		return int(default)
	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")

def load_path_config(config_path: Path, field_name: str, default: Path | None = None) -> Path:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if separator and key.strip() == field_name:
			parsed = value.strip()
			if not parsed:
				raise ValueError(f"Missing value for '{field_name}' in: {config_path}")
			if parsed[:1] in {'"', "'"} and parsed[-1:] == parsed[:1]:
				parsed = parsed[1:-1]
			return Path(parsed)

	if default is not None:
		return default
	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")

def _parse_float_list(value: str, expected_len: int, context: str) -> np.ndarray:
	value = value.strip()
	if not (value.startswith("[") and value.endswith("]")):
		raise ValueError(f"Invalid list for {context}: {value}")
	parts = [item.strip() for item in value[1:-1].split(",") if item.strip()]
	if len(parts) != expected_len:
		raise ValueError(f"{context} must contain exactly {expected_len} values: {value}")
	return np.asarray([float(item) for item in parts], dtype=np.float32)

def load_key_frame_config(
	config_path: Path,
	section_name: str = "key_frame_config",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	offset_map: dict[str, np.ndarray] = {}
	axis_map: dict[str, np.ndarray] = {}
	in_section = False
	current_body: str | None = None
	in_axis_subsection = False
	current_axes: dict[str, np.ndarray] = {}
	current_offset: np.ndarray | None = None

	def flush_current() -> None:
		nonlocal current_body, in_axis_subsection, current_axes, current_offset
		if current_body is None:
			return
		if current_offset is None:
			current_offset = np.zeros(3, dtype=np.float32)
		if set(current_axes.keys()) != {"x", "y", "z"}:
			current_axes = {
				"x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
				"y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
				"z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
			}
		offset_map[current_body] = current_offset
		axis_map[current_body] = np.column_stack(
			[current_axes["x"], current_axes["y"], current_axes["z"]]
		).astype(np.float32)
		current_body = None
		in_axis_subsection = False
		current_axes = {}
		current_offset = None

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].rstrip()
		if not line.strip():
			continue

		if not in_section:
			if line.strip() == f"{section_name}:":
				in_section = True
			continue

		if not raw_line[:1].isspace():
			break

		indent = len(raw_line) - len(raw_line.lstrip())
		stripped = line.strip()
		if indent <= 2 and ":" in stripped:
			flush_current()
			name, _, _rest = stripped.partition(":")
			current_body = name.strip()
			continue

		if current_body is None:
			continue

		if indent <= 4 and stripped.startswith("offset_deg_xyz"):
			_key, _, value = stripped.partition(":")
			current_offset = _parse_float_list(value, expected_len=3, context=f"{current_body}.offset_deg_xyz")
			continue

		if indent <= 4 and stripped.startswith("axis_map_cols"):
			in_axis_subsection = True
			continue

		if in_axis_subsection and indent >= 6 and ":" in stripped:
			axis_name, _, value = stripped.partition(":")
			axis_name = axis_name.strip().lower()
			if axis_name not in {"x", "y", "z"}:
				continue
			current_axes[axis_name] = _parse_float_list(
				value,
				expected_len=3,
				context=f"{current_body}.axis_map_cols.{axis_name}",
			)
			continue
		else:
			continue

	flush_current()

	if not offset_map:
		raise ValueError(f"No {section_name} entries found in: {config_path}")
	return offset_map, axis_map

def compute_robot_link_lengths(
	config_path: Path,
	robot_mjcf_path: Path,
) -> dict[str, float]:
	robot_links = load_robot_links_config(config_path)
	if not robot_mjcf_path.exists():
		raise FileNotFoundError(f"Robot MJCF file not found: {robot_mjcf_path}")

	model = mujoco.MjModel.from_xml_path(str(robot_mjcf_path))
	data = mujoco.MjData(model)
	mujoco.mj_forward(model, data)

	link_lengths: dict[str, float] = {}
	for link_name, (start_body, end_body) in robot_links.items():
		start_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, start_body)
		if start_body_id < 0:
			raise ValueError(f"Missing robot body in MJCF: {start_body}")
		end_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)
		if end_body_id < 0:
			raise ValueError(f"Missing robot body in MJCF: {end_body}")

		start_pos = data.xpos[start_body_id]
		end_pos = data.xpos[end_body_id]
		link_lengths[link_name] = float(np.linalg.norm(end_pos - start_pos))

	return link_lengths

def compute_robot_body_local_offset(
	robot_mjcf_path: Path,
	anchor_body: str,
	target_body: str,
) -> np.ndarray:
	if not robot_mjcf_path.exists():
		raise FileNotFoundError(f"Robot MJCF file not found: {robot_mjcf_path}")

	model = mujoco.MjModel.from_xml_path(str(robot_mjcf_path))
	data = mujoco.MjData(model)
	mujoco.mj_forward(model, data)

	anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body)
	if anchor_body_id < 0:
		raise ValueError(f"Missing robot anchor body in MJCF: {anchor_body}")
	target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
	if target_body_id < 0:
		raise ValueError(f"Missing robot target body in MJCF: {target_body}")

	anchor_pos = data.xpos[anchor_body_id]
	target_pos = data.xpos[target_body_id]
	anchor_rot = data.xmat[anchor_body_id].reshape(3, 3)
	local_offset = anchor_rot.T @ (target_pos - anchor_pos)
	return local_offset.astype(np.float32)

def quat_rotate_vectors_wxyz(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
	vectors = np.broadcast_to(vector.astype(np.float32), (quaternions.shape[0], 3))
	rotated = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).apply(vectors)
	return rotated.astype(np.float32)

def compute_windowed_point_speeds(
	point_positions: np.ndarray,
	fps: float,
	window: int,
	progress_desc: str | None = None,
) -> np.ndarray:
	if fps <= 0.0:
		raise ValueError(f"FPS must be positive, got {fps}")
	if window <= 0:
		raise ValueError(f"Window must be positive, got {window}")

	num_frames = point_positions.shape[0]
	half_window = max(1, window // 2)
	speeds = np.zeros(point_positions.shape[:2], dtype=np.float32)

	frame_iter = range(num_frames)
	if progress_desc is not None:
		frame_iter = iter_progress(frame_iter, total=num_frames, desc=progress_desc, unit="frame")
	for frame_idx in frame_iter:
		start_idx = max(0, frame_idx - half_window)
		end_idx = min(num_frames - 1, frame_idx + half_window)
		frame_delta = end_idx - start_idx
		if frame_delta <= 0:
			continue
		displacement = point_positions[end_idx, :, :] - point_positions[start_idx, :, :]
		speeds[frame_idx, :] = np.linalg.norm(displacement, axis=-1) / (frame_delta / fps)

	return speeds

def resolve_contact_height_reference_indices(
	keypoint_names: list[str],
	contact_names: tuple[str, ...],
) -> np.ndarray:
	keypoint_index = {name: idx for idx, name in enumerate(keypoint_names)}
	resolved_indices = np.full(len(contact_names), -1, dtype=np.int32)
	for contact_idx, contact_name in enumerate(contact_names):
		candidate_names = (contact_name, f"{contact_name}_link")
		for candidate_name in candidate_names:
			if candidate_name in keypoint_index:
				resolved_indices[contact_idx] = keypoint_index[candidate_name]
				break
	return resolved_indices

def apply_low_pass_filter(values: np.ndarray, alpha: float) -> np.ndarray:
	if values.ndim != 1:
		raise ValueError(f"values shape must be [T], got {values.shape}")
	if not (0.0 < float(alpha) <= 1.0):
		raise ValueError(f"Low-pass alpha must be in (0, 1], got {alpha}")

	filtered = values.astype(np.float32).copy()
	for frame_idx in range(1, filtered.shape[0]):
		filtered[frame_idx] = (
			float(alpha) * filtered[frame_idx]
			+ (1.0 - float(alpha)) * filtered[frame_idx - 1]
		)
	return filtered

def compute_contact_height_offsets(
	keypoints: np.ndarray,
	keypoint_names: list[str],
	contact_names: tuple[str, ...],
	contact_positions: np.ndarray,
	contact_states: np.ndarray,
	height_lpf_alpha: float = 1.0,
) -> np.ndarray:
	if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
		raise ValueError(f"keypoints shape must be [T, K, 3], got {keypoints.shape}")
	if contact_positions.ndim != 3 or contact_positions.shape[-1] != 3:
		raise ValueError(
			"contact_positions shape must be [T, C, 3], got "
			f"{contact_positions.shape}"
		)
	if contact_states.shape != contact_positions.shape[:2]:
		raise ValueError(
			"contact_states shape must match contact_positions[:2], got "
			f"{contact_states.shape} and {contact_positions.shape[:2]}"
		)
	if keypoints.shape[0] != contact_positions.shape[0]:
		raise ValueError(
			"keypoints and contact_positions must have the same frame count, got "
			f"{keypoints.shape[0]} and {contact_positions.shape[0]}"
		)

	contact_keypoint_indices = resolve_contact_height_reference_indices(keypoint_names, contact_names)
	height_offsets = np.zeros(contact_positions.shape[0], dtype=np.float32)
	last_height = 0.0
	for frame_idx in range(contact_positions.shape[0]):
		active_mask = np.asarray(contact_states[frame_idx], dtype=np.bool_)
		if not np.any(active_mask):
			height_offsets[frame_idx] = last_height
			continue

		active_contact_indices = np.flatnonzero(active_mask)
		active_heights: list[float] = []
		for contact_idx in active_contact_indices:
			keypoint_idx = int(contact_keypoint_indices[contact_idx])
			if keypoint_idx >= 0:
				active_heights.append(float(keypoints[frame_idx, keypoint_idx, 2]))
			else:
				active_heights.append(float(contact_positions[frame_idx, contact_idx, 2]))

		last_height = min(active_heights)
		height_offsets[frame_idx] = last_height
	if height_offsets.shape[0] <= 1 or float(height_lpf_alpha) >= 1.0:
		return height_offsets
	return apply_low_pass_filter(height_offsets, alpha=height_lpf_alpha)

def compute_link_geometry_from_positions(
	link_pairs: dict[str, tuple[str, str]],
	body_slots: dict[str, int],
	positions: np.ndarray,
) -> tuple[
		dict[str, np.ndarray],
		dict[str, np.ndarray],
		dict[str, tuple[str, str]],
]:
	link_lengths: dict[str, np.ndarray] = {}
	link_vectors: dict[str, np.ndarray] = {}
	resolved_pairs: dict[str, tuple[str, str]] = {}

	for link_name, (parent_body, child_body) in link_pairs.items():
		if parent_body not in body_slots:
			raise ValueError(f"Missing link parent body in replay buffers: {parent_body}")
		if child_body not in body_slots:
			raise ValueError(f"Missing link child body in replay buffers: {child_body}")

		parent_idx = body_slots[parent_body]
		child_idx = body_slots[child_body]
		link_vector = positions[:, child_idx, :] - positions[:, parent_idx, :]
		link_length = np.linalg.norm(link_vector, axis=-1)
		if np.any(link_length <= 1e-8):
			raise ValueError(f"Skeleton link has near-zero length for at least one frame: {link_name}")

		link_lengths[link_name] = link_length.astype(np.float32)
		link_vectors[link_name] = link_vector.astype(np.float32)
		resolved_pairs[link_name] = (parent_body, child_body)
		# link_is_static[link_name] = bool(np.allclose(link_length, link_length[0], rtol=1e-5, atol=1e-6))

	return link_lengths, link_vectors, resolved_pairs

def compute_link_scale_factors(
	robot_link_lengths: dict[str, float],
	skeleton_link_lengths: dict[str, np.ndarray],
) -> tuple[dict[str, float | np.ndarray], dict[str, bool]]:
	link_scales: dict[str, float | np.ndarray] = {}
	link_scale_is_static: dict[str, bool] = {}

	for link_name, skeleton_length in skeleton_link_lengths.items():
		if link_name not in robot_link_lengths:
			raise ValueError(f"Missing robot link length for skeleton link: {link_name}")

		scale = robot_link_lengths[link_name] / skeleton_length
		link_scales[link_name] = scale.astype(np.float32)
		link_scale_is_static[link_name] = False

	return link_scales, link_scale_is_static

def compute_leg_displacement_scale(
	robot_link_lengths: dict[str, float],
	skeleton_link_vectors: dict[str, np.ndarray],
	knee_angle_offset_degrees: float,
) -> tuple[float, float, float]:
	required_robot_links = ("left_thigh", "right_thigh", "left_calf", "right_calf")
	for link_name in required_robot_links:
		if link_name not in robot_link_lengths:
			raise ValueError(f"Missing robot leg link length: {link_name}")

	required_skeleton_links = ("left_thigh", "right_thigh", "left_calf", "right_calf")
	for link_name in required_skeleton_links:
		if link_name not in skeleton_link_vectors:
			raise ValueError(f"Missing skeleton leg link vector: {link_name}")

	robot_thigh_length = 0.5 * (
		float(robot_link_lengths["left_thigh"]) + float(robot_link_lengths["right_thigh"])
	)
	robot_calf_length = 0.5 * (
		float(robot_link_lengths["left_calf"]) + float(robot_link_lengths["right_calf"])
	)
	robot_leg_length = float(
		np.sqrt(
			max(
				robot_thigh_length**2
				+ robot_calf_length**2
				+ 2.0
				* robot_thigh_length
				* robot_calf_length
				* np.cos(np.radians(float(knee_angle_offset_degrees))),
				0.0,
			)
		)
	)

	left_skeleton_leg_length = np.linalg.norm(skeleton_link_vectors["left_thigh"], axis=-1) + np.linalg.norm(
		skeleton_link_vectors["left_calf"], axis=-1
	)
	right_skeleton_leg_length = np.linalg.norm(skeleton_link_vectors["right_thigh"], axis=-1) + np.linalg.norm(
		skeleton_link_vectors["right_calf"], axis=-1
	)
	skeleton_leg_length = float(np.mean(0.5 * (left_skeleton_leg_length + right_skeleton_leg_length)))
	if skeleton_leg_length <= 1e-8:
		raise ValueError(f"Skeleton leg length must be positive, got {skeleton_leg_length}")

	leg_displacement_scale = float(robot_leg_length / skeleton_leg_length)
	return leg_displacement_scale, robot_leg_length, skeleton_leg_length

def scale_keypoint_frame_displacements(
	keypoints: np.ndarray,
	displacement_scale: float | np.ndarray,
	root_keypoint_idx: int = 0,
) -> np.ndarray:
	if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
		raise ValueError(f"keypoints shape must be [T, K, 3], got {keypoints.shape}")
	scale = np.asarray(displacement_scale, dtype=np.float32)
	if scale.ndim == 0:
		scale = np.repeat(scale, 3)
	if scale.shape != (3,):
		raise ValueError(f"displacement_scale must be a scalar or length-3 vector, got {scale.shape}")
	if np.any(scale <= 0.0):
		raise ValueError(f"displacement_scale must be positive, got {scale}")
	if root_keypoint_idx < 0 or root_keypoint_idx >= keypoints.shape[1]:
		raise ValueError(
			f"root_keypoint_idx must be within [0, {keypoints.shape[1]}), got {root_keypoint_idx}"
		)
	if keypoints.shape[0] <= 1 or np.allclose(scale, 1.0):
		return keypoints.astype(np.float32, copy=True)

	root_positions = keypoints[:, root_keypoint_idx, :].astype(np.float32)
	root_frame_deltas = np.diff(root_positions, axis=0)
	scaled_root_positions = np.empty_like(root_positions, dtype=np.float32)
	scaled_root_positions[0] = root_positions[0]
	scaled_root_positions[1:] = scaled_root_positions[0:1] + np.cumsum(
		(scale * root_frame_deltas).astype(np.float32),
		axis=0,
	)
	translation_offsets = scaled_root_positions - root_positions
	return (keypoints + translation_offsets[:, None, :]).astype(np.float32)

def load_root_displacement_scale(
	config_path: Path,
	leg_displacement_scale: float,
) -> tuple[np.ndarray, tuple[str, str, str]]:
	"""Resolve per-axis root displacement scaling from a robot config.

	Each world axis may either preserve the source trajectory or scale it by the
	target/source leg-length ratio. Missing configuration keeps the historical
	uniform leg-length scaling behavior.
	"""
	with config_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	raw_modes = config.get(
		"root_displacement_scaling",
		{"x": "leg_length", "y": "leg_length", "z": "leg_length"},
	)
	if not isinstance(raw_modes, dict):
		raise ValueError(f"root_displacement_scaling must be a mapping in: {config_path}")

	modes: list[str] = []
	scales: list[float] = []
	for axis in ("x", "y", "z"):
		mode = raw_modes.get(axis, "leg_length")
		if mode == "preserve_source":
			scale = 1.0
		elif mode == "leg_length":
			scale = float(leg_displacement_scale)
		else:
			raise ValueError(
				f"root_displacement_scaling.{axis} must be 'preserve_source' or "
				f"'leg_length', got {mode!r} in: {config_path}"
			)
		modes.append(mode)
		scales.append(scale)
	return np.asarray(scales, dtype=np.float32), tuple(modes)

def apply_link_scales_to_positions(
	positions: np.ndarray,
	link_pairs: dict[str, tuple[str, str]],
	body_slots: dict[str, int],
	link_scales: dict[str, float | np.ndarray],
	link_vectors: dict[str, np.ndarray],
	knee_angle_offset_degrees: float,
	left_leg_links: tuple[str, str, str],
	right_leg_links: tuple[str, str, str],
	quaternions: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
	scaled_positions = positions.copy()
	scaled_quaternions = quaternions.copy() if quaternions is not None else None
	for link_name, (parent_body, child_body) in link_pairs.items():
		parent_idx = body_slots[parent_body]
		child_idx = body_slots[child_body]
		scale = link_scales[link_name]
		if np.isscalar(scale):
			scaled_positions[:, child_idx, :] = (
				scaled_positions[:, parent_idx, :] + float(scale) * link_vectors[link_name]
			)
		else:
			scaled_positions[:, child_idx, :] = (
				scaled_positions[:, parent_idx, :] + scale[:, None] * link_vectors[link_name]
			)

	# Blend knee joint by increasing knee angle and only updating foot keypoints.
	blend_offset_radius = np.radians(float(knee_angle_offset_degrees))

	for body_name in left_leg_links + right_leg_links:
		if body_name not in body_slots:
			raise ValueError(f"Missing leg body in replay buffers: {body_name}")

	left_hip_idx = body_slots[left_leg_links[0]]
	left_knee_idx = body_slots[left_leg_links[1]]
	left_foot_idx = body_slots[left_leg_links[2]]
	left_hip_to_knee = scaled_positions[:, left_knee_idx, :] - scaled_positions[:, left_hip_idx, :]
	left_knee_to_foot = scaled_positions[:, left_foot_idx, :] - scaled_positions[:, left_knee_idx, :]
	left_hip_to_knee_length = np.linalg.norm(left_hip_to_knee, axis=-1)
	left_knee_to_foot_length = np.linalg.norm(left_knee_to_foot, axis=-1)

	right_hip_idx = body_slots[right_leg_links[0]]
	right_knee_idx = body_slots[right_leg_links[1]]
	right_foot_idx = body_slots[right_leg_links[2]]
	right_hip_to_knee = scaled_positions[:, right_knee_idx, :] - scaled_positions[:, right_hip_idx, :]
	right_knee_to_foot = scaled_positions[:, right_foot_idx, :] - scaled_positions[:, right_knee_idx, :]
	right_hip_to_knee_length = np.linalg.norm(right_hip_to_knee, axis=-1)
	right_knee_to_foot_length = np.linalg.norm(right_knee_to_foot, axis=-1)

	left_hip_to_foot = scaled_positions[:, left_foot_idx, :] - scaled_positions[:, left_hip_idx, :]
	right_hip_to_foot = scaled_positions[:, right_foot_idx, :] - scaled_positions[:, right_hip_idx, :]
	left_hip_to_foot_length = np.linalg.norm(left_hip_to_foot, axis=-1)
	right_hip_to_foot_length = np.linalg.norm(right_hip_to_foot, axis=-1)
	left_hip_to_foot_dir_norm = left_hip_to_foot / np.clip(left_hip_to_foot_length[:, None], 1e-8, None)
	right_hip_to_foot_dir_norm = right_hip_to_foot / np.clip(right_hip_to_foot_length[:, None], 1e-8, None)

	left_cos = np.sum(left_hip_to_knee * left_knee_to_foot, axis=-1) / np.clip(
		left_hip_to_knee_length * left_knee_to_foot_length,
		1e-8,
		None,
	)
	right_cos = np.sum(right_hip_to_knee * right_knee_to_foot, axis=-1) / np.clip(
		right_hip_to_knee_length * right_knee_to_foot_length,
		1e-8,
		None,
	)
	current_angle_left = np.arccos(np.clip(left_cos, -1.0, 1.0))
	blended_angle_left = current_angle_left + blend_offset_radius
	current_angle_right = np.arccos(np.clip(right_cos, -1.0, 1.0))
	blended_angle_right = current_angle_right + blend_offset_radius

	left_hip_to_foot_blended_length = np.sqrt(
		np.clip(
			left_hip_to_knee_length**2
			+ left_knee_to_foot_length**2
			+ 2 * left_hip_to_knee_length * left_knee_to_foot_length * np.cos(blended_angle_left),
			0.0,
			None,
		)
	)
	right_hip_to_foot_blended_length = np.sqrt(
		np.clip(
			right_hip_to_knee_length**2
			+ right_knee_to_foot_length**2
			+ 2 * right_hip_to_knee_length * right_knee_to_foot_length * np.cos(blended_angle_right),
			0.0,
			None,
		)
	)

	left_blended_foot_pos = (
		scaled_positions[:, left_hip_idx, :]
		+ left_hip_to_foot_dir_norm * left_hip_to_foot_blended_length[:, None]
	)
	right_blended_foot_pos = (
		scaled_positions[:, right_hip_idx, :]
		+ right_hip_to_foot_dir_norm * right_hip_to_foot_blended_length[:, None]
	)
	left_blended_knee_pos = compute_two_bone_ik_knee_positions(
		hip_positions=scaled_positions[:, left_hip_idx, :],
		knee_positions=scaled_positions[:, left_knee_idx, :],
		foot_positions=scaled_positions[:, left_foot_idx, :],
		target_foot_positions=left_blended_foot_pos,
	)
	right_blended_knee_pos = compute_two_bone_ik_knee_positions(
		hip_positions=scaled_positions[:, right_hip_idx, :],
		knee_positions=scaled_positions[:, right_knee_idx, :],
		foot_positions=scaled_positions[:, right_foot_idx, :],
		target_foot_positions=right_blended_foot_pos,
	)

	scaled_positions[:, left_foot_idx, :] = left_blended_foot_pos
	scaled_positions[:, right_foot_idx, :] = right_blended_foot_pos
	scaled_positions[:, left_knee_idx, :] = left_blended_knee_pos
	scaled_positions[:, right_knee_idx, :] = right_blended_knee_pos

	if scaled_quaternions is not None:
		# Rotate the hip and knee orientations to follow the bent leg, while keeping
		# the foot orientation untouched. The hip body follows the thigh segment
		# (hip -> knee) and the knee body follows the shank segment (knee -> foot).
		# Each rotation is about the normal of the hip/knee/foot plane (the cross
		# product of the original and blended segment directions).
		left_hip_to_knee_blended = left_blended_knee_pos - scaled_positions[:, left_hip_idx, :]
		left_knee_to_foot_blended = left_blended_foot_pos - left_blended_knee_pos
		right_hip_to_knee_blended = right_blended_knee_pos - scaled_positions[:, right_hip_idx, :]
		right_knee_to_foot_blended = right_blended_foot_pos - right_blended_knee_pos

		left_hip_delta = batch_rotation_between_vectors_wxyz(left_hip_to_knee, left_hip_to_knee_blended)
		left_knee_delta = batch_rotation_between_vectors_wxyz(left_knee_to_foot, left_knee_to_foot_blended)
		right_hip_delta = batch_rotation_between_vectors_wxyz(right_hip_to_knee, right_hip_to_knee_blended)
		right_knee_delta = batch_rotation_between_vectors_wxyz(right_knee_to_foot, right_knee_to_foot_blended)

		scaled_quaternions[:, left_hip_idx, :] = multiply_quaternions_wxyz(
			left_hip_delta, scaled_quaternions[:, left_hip_idx, :]
		)
		scaled_quaternions[:, left_knee_idx, :] = multiply_quaternions_wxyz(
			left_knee_delta, scaled_quaternions[:, left_knee_idx, :]
		)
		scaled_quaternions[:, right_hip_idx, :] = multiply_quaternions_wxyz(
			right_hip_delta, scaled_quaternions[:, right_hip_idx, :]
		)
		scaled_quaternions[:, right_knee_idx, :] = multiply_quaternions_wxyz(
			right_knee_delta, scaled_quaternions[:, right_knee_idx, :]
		)
		return scaled_positions, scaled_quaternions

	return scaled_positions

def average_quaternions_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	# Align hemisphere before averaging to avoid cancellation.
	dot = np.sum(q1 * q2, axis=-1, keepdims=True)
	q2_aligned = np.where(dot < 0.0, -q2, q2)
	q_avg = q1 + q2_aligned
	q_avg /= np.clip(np.linalg.norm(q_avg, axis=-1, keepdims=True), 1e-8, None)
	return q_avg.astype(np.float32)

def select_frame_slice(num_frames: int, start_frame: int, end_frame: int, stride: int) -> np.ndarray:
	start = max(0, start_frame)
	stop = num_frames if end_frame < 0 else min(num_frames, end_frame + 1)
	if start >= stop:
		raise ValueError(f"Empty frame range: start={start_frame}, end={end_frame}, total={num_frames}")
	if stride <= 0:
		raise ValueError(f"Stride must be positive, got {stride}")
	return np.arange(start, stop, stride, dtype=np.int32)

def slice_frame_aligned_arrays(
	frame_ids: np.ndarray,
	**arrays: np.ndarray | None,
) -> dict[str, np.ndarray | None]:
	"""Select the same frames from synchronized motion arrays.

	The replay CLIs expose one frame range for visualization and export.  Keeping
	the selection in one helper prevents a sliced video from silently producing a
	full-length keypoint payload.
	"""
	if frame_ids.ndim != 1 or frame_ids.size == 0:
		raise ValueError("frame_ids must be a non-empty 1D array")
	if np.any(frame_ids < 0):
		raise ValueError("frame_ids must be non-negative")
	last_frame = int(frame_ids[-1])
	selected: dict[str, np.ndarray | None] = {}
	for name, values in arrays.items():
		if values is None:
			selected[name] = None
			continue
		if values.ndim == 0:
			raise ValueError(f"{name} must have a frame dimension")
		if values.shape[0] <= last_frame:
			raise ValueError(
				f"{name} has {values.shape[0]} frames but selection includes {last_frame}"
			)
		selected[name] = values[frame_ids]
	return selected

def save_keypoints_pkl(
	output_path: Path,
	keypoint_names: list[str],
	positions: np.ndarray,
	quaternions: np.ndarray,
	fps: float | None = None,
	contact_names: tuple[str, ...] | None = None,
	contact_positions: np.ndarray | None = None,
	contact_speeds: np.ndarray | None = None,
	contact_states: np.ndarray | None = None,
	contact_vel_window: int | None = None,
	contact_vel_threshold: float | None = None,
	contact_height_threshold: float | None = None,
	source_root_positions: np.ndarray | None = None,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"keypoint_names": keypoint_names,
		"positions": positions.astype(np.float32),
		"quaternions": quaternions.astype(np.float32),
	}
	if fps is not None:
		payload["fps"] = float(fps)
	if contact_names is not None:
		payload["contact_names"] = list(contact_names)
	# if contact_positions is not None:
	# 	payload["contact_positions"] = contact_positions.astype(np.float32)
	# if contact_speeds is not None:
	# 	payload["contact_speeds"] = contact_speeds.astype(np.float32)
	if contact_states is not None:
		payload["contact_states"] = contact_states.astype(np.bool_)
	if source_root_positions is not None:
		source_root_positions = np.asarray(source_root_positions, dtype=np.float32)
		if source_root_positions.shape != (positions.shape[0], 3):
			raise ValueError(
				"source_root_positions must have shape "
				f"({positions.shape[0]}, 3), got {source_root_positions.shape}"
			)
		payload["source_root_positions"] = source_root_positions
	# if contact_vel_window is not None:
	# 	payload["contact_vel_window"] = int(contact_vel_window)
	# if contact_vel_threshold is not None:
	# 	payload["contact_vel_threshold"] = float(contact_vel_threshold)
	# if contact_height_threshold is not None:
	# 	payload["contact_height_threshold"] = float(contact_height_threshold)
	with output_path.open("wb") as f:
		pickle.dump(payload, f)
	
	print(f"Saved keypoints to: {output_path}")
