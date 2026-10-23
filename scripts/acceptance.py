"""Fixed R9 publication gate. No caller may override these thresholds."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARD = json.loads((ROOT / 'validation/acceptance.json').read_text())


def validate_kkt(kkt):
    for name in ('stationarity', 'complementarity'):
        if kkt.get(name + '_normalization') != STANDARD[name + '_normalization']:
            raise ValueError(f'Wrong {name} normalization')
        value = float(kkt[name])
        if not math.isfinite(value) or value > STANDARD[name + '_max'] or value < 0:
            raise ValueError(f'{name} exceeds the fixed R9 gate: {value}')
    value = float(kkt['violation'])
    if not math.isfinite(value) or not 0 <= value <= STANDARD['constraint_violation_max']:
        raise ValueError('Constraint violation exceeds the fixed R9 gate')


def validate_record(record):
    if record['status'] != 'accepted' or not record['trajectory_converged']:
        raise ValueError('Motion was not accepted by the trajectory solver')
    if record['frames'] != record['expected_frames'] or record['fps'] != STANDARD['fps']:
        raise ValueError('Incomplete motion or changed timing')
    validate_kkt(record['final_kkt'])
    for field in ('minimum_ground_gap_m', 'minimum_self_clearance_m'):
        value = float(record[field])
        if not math.isfinite(value) or value < STANDARD[field]:
            raise ValueError(f'{field} fails the geometric audit')
    for field in ('joint_limit_violation_rad', 'root_xy_error_m'):
        value = float(record[field])
        if not math.isfinite(value) or value > STANDARD[field + '_max']:
            raise ValueError(f'{field} exceeds its bound')
