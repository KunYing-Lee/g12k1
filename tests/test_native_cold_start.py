"""The exact native kernel must build from the repository with an empty cache."""
import os
from pathlib import Path
import subprocess
import sys


def test_cold_native_build_from_repository(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q',
         'tests/test_convex_cylinder_distance.py::test_face_cap_and_rim_distances_have_analytic_values'],
        cwd=root,
        env=dict(os.environ, TMPDIR=str(tmp_path), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.rglob('*.so')), 'The test did not create a fresh native extension'
