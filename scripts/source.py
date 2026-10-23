"""Parse Bones G1 degrees/cm CSV using the certified R9 conversion."""
from __future__ import annotations
import csv, math
from pathlib import Path

def euler_xyz_deg_to_quat_xyzw(rx_deg: float, ry_deg: float, rz_deg: float) -> tuple[float, float, float, float]:
    """Convert intrinsic XYZ Euler angles (deg) to quaternion in XYZW order."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)

    # wxyz for intrinsic XYZ
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return x, y, z, w

def safe_float(row: dict[str, str], key: str) -> float:
    """Read required finite observations; never synthesize a missing angle."""
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required motion value: {key}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite motion value: {key}")
    return result

def convert_one_file(src: Path, dst: Path, root_scale: float) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {src}")

        joint_cols = [c for c in reader.fieldnames if c.endswith("_dof")]

        frames = 0
        with dst.open("w", newline="") as f_out:
            writer = csv.writer(f_out)
            for row in reader:
                tx = safe_float(row, "root_translateX") * root_scale
                ty = safe_float(row, "root_translateY") * root_scale
                tz = safe_float(row, "root_translateZ") * root_scale

                qx, qy, qz, qw = euler_xyz_deg_to_quat_xyzw(
                    safe_float(row, "root_rotateX"),
                    safe_float(row, "root_rotateY"),
                    safe_float(row, "root_rotateZ"),
                )

                out_row: list[float] = [tx, ty, tz, qx, qy, qz, qw]
                for jc in joint_cols:
                    out_row.append(math.radians(safe_float(row, jc)))

                writer.writerow([f"{v:.6f}" for v in out_row])
                frames += 1

    return frames, len(joint_cols)
