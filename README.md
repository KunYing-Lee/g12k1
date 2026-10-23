# g12k1 — fixed R9 G1 → Booster K1 retargeting

A small, headless extraction of the validated R9 Bones G1-to-K1 pipeline. It reads the native Bones G1 CSV format, fits the entire motion to the serial 22-DoF K1 model, and publishes a motion only after the original numerical and geometric acceptance checks pass. It does not contain SMPL processing, other robots, old command-line interfaces, training code, datasets, or GPU prototypes.

## Reproduce the environment

Production is Linux x86-64, CPython **3.14.6**. A C compiler is required for the float64 exact collision kernels. MuJoCo models and every referenced mesh are included.

```sh
uv python install 3.14.6
uv venv --python 3.14.6
uv pip sync requirements-test.lock
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

`requirements.lock` pins all runtime packages; the test lock adds only pytest dependencies. On macOS, frame parallelism is unsupported by the current fork-based implementation; use `RETARGET_FRAME_WORKERS=1` for development. Production equivalence is certified on Linux.

`r9.0.1` explicitly disables Python-package autodiscovery for this script-based repository. This fixes first-use CFFI compilation when no native cache exists; solver files, dependencies, acceptance and the numerical producer fingerprint are identical to `r9.0.0`. The cold-cache analytic geometry test reproduces the previous failure and passes with this metadata fix; all 63 Linux tests pass.

## Input and output

An input manifest is a JSON array. Each entry has a unique `member`, an absolute source `path`, its `sha256`, and optionally its full `frames` count:

```json
[{"member":"g1/csv/session/motion.csv","path":"/data/g1/csv/session/motion.csv","sha256":"<64 hexadecimal characters>","frames":759}]
```

Source CSVs contain the original root translation in centimetres, root Euler angles and ordered G1 joint angles in degrees. Missing or nonfinite input values and unexpected G1 joint-column order are errors. No motion is subsampled, truncated, or split across machines.

Outputs under `motion/<member>` have no header: position XYZ in metres, quaternion **XYZW**, then K1 joint angles in model order, in radians. The full **120 Hz** source timing is retained. `status/` records source/output/producer hashes, geometric audits, convergence certificates and timings. `diagnostics/` contains compressed solver histories. Failed clips never publish a new accepted motion.

## Run

Validation runs use the same strict per-motion gate as production:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RETARGET_FRAME_WORKERS=2 \
  .venv/bin/python scripts/run.py --manifest /data/regression.json \
  --output runs/regression --workers 8 --frame-budget 12000 --validation
```

To reproduce the release certificate from original R9 reference outputs and the extracted implementation's outputs, use `tools/certify.py` with the original 45-motion manifest. It verifies both producers, input/output hashes, every acceptance record, and exact numeric equality:

```sh
.venv/bin/python tools/certify.py --manifest /data/regression45.json \
  --reference-output /data/original-r9-results --candidate-output runs/regression \
  --output validation/release.json
```

Production additionally requires a matching 45-motion release certificate:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RETARGET_FRAME_WORKERS=2 \
  .venv/bin/python scripts/run.py --manifest /data/assigned-shard.json \
  --output runs/production --workers 8 --frame-budget 12000 \
  --gate validation/release.json
```

`workers` means concurrent complete motions; `RETARGET_FRAME_WORKERS` is the number of frame workers inside each motion. The frame budget bounds admitted work, not motion length or solver precision. A motion larger than the budget runs alone with six frame workers. SIGTERM/SIGINT stops admission and drains active clips. A file lock prevents two managers from owning the same output directory. A restart verifies accepted output hashes before resuming.

## Original acceptance standard

`validation/acceptance.json` is the machine-readable contract, enforced by `scripts/acceptance.py`:

- Float64 nonlinear solver, FK/Jacobians, collision queries and QPs; complete finite output.
- Per-frame normalized KKT stationarity ≤ **1e-6**; absolute complementarity ≤ **1e-6**.
- Aggregate constraint violation ≤ **1e-8**.
- Independently evaluated ground gap ≥ **−1e-8 m** and self-clearance ≥ **0.001 − 1e-8 m**.
- Joint-limit violation and horizontal root-trajectory error ≤ **1e-8**.
- Source-relative velocity objective `D(q − source)`, unchanged weights and iteration limits.

There is no output smoothing, post-solve ground projection, or acceptance relaxation. R9's pre-existing target-generation representation is preserved exactly: the intermediate converted CSV uses six decimal places, target arrays use float32, and source-contact height registration uses the existing target filter. These are upstream observation construction, **not a claim of an end-to-end float64 input pipeline**. Changing them would require a separate algorithm version and validation; this extraction does not change them.

These are kinematic/numerical acceptance conditions, not a physical controller or dynamic feasibility certificate. The known reduced dance-arm amplitude is an objective tradeoff retained from R9; this release does not claim to fix it or establish convergence for every motion in the full dataset.

## Distributed ownership and concurrency tests

Drain old managers before producing the verified completed-artifact ledger. Run `scripts/partition.py` once with the immutable full manifest and positive integer host weights. It writes disjoint host manifests, preserves the accepted artifacts separately, and checks exact coverage. Remap source `path` fields for each host without changing `member` or `sha256`. Never point multiple hosts at the full unpartitioned manifest. Do not reassign a shard while its owner is running.

```sh
.venv/bin/python scripts/partition.py --manifest /data/full.json \
  --completed /data/verified-completed.json --hosts '{"alpha":1,"g1":4}' \
  --output /data/ownership-plan
.venv/bin/python tools/benchmark.py --manifest /data/benchmark96.json \
  --gate validation/release.json --output runs/concurrency
```

The benchmark runs `16×2, 32×2, 48×2, 48×2, 32×2, 16×2` on identical complete motions, with a common 72,000-frame admission budget. It checks every acceptance record, output checksum and exact numerical agreement across configurations. Selection uses median total completion time, including process startup, preparation, solving, validation and publication. Resource samples include container CPU/memory pressure.

## Measured g1 concurrency

The six-run complete-motion benchmark accepted **576/576 runs** and produced exactly equal motion arrays across configurations. Each run covered **96 clips / 65,768 frames**. The numerical algorithm is fixed by `r9.0.0`; production uses `r9.0.1`, which only corrects native-extension build metadata.

| Clip workers × frame workers | First run (s) | Reverse run (s) | Median (s) | Clips/hour |
|---|---:|---:|---:|---:|
| 16×2 | 830.7 | 830.7 | 830.7 | 416.0 |
| 32×2 | 640.6 | 640.6 | 640.6 | 539.5 |
| 48×2 | 585.5 | 585.5 | 585.5 | 590.3 |

Selected **48×2** by median completion time. See `validation/concurrency_g1.json` for raw timings, identity and container resource counters. These are measurements on this sample, not a full-dataset ETA. Static ownership excludes already accepted original, regression and benchmark artifacts; alpha remains at 8×2.

During the original R9 full run, `warm_up_legs_R_002__A361_M` failed to converge in block 960:1088 after 200 iterations (stationarity 0.0664). It was not published as accepted. This is retained in `validation/known_failures.json`; the fixed release does not claim universal convergence or change the acceptance standard.

## Code map

- `source.py`: strict Bones input conversion.
- `targets.py`, `kinematics.py`: G1 FK, morphology mapping and Cartesian/contact target generation.
- `solver.py`: constrained per-frame initialization, exact body-ground/self-collision constraints and nonlinear SQP.
- `trajectory.py`: complete-motion constrained trajectory fitting and parallel block solves.
- `geometry.py`: exact float64 polytope/cylinder geometry and witness Jacobians.
- `batch.py`, `acceptance.py`: artifact identity, audit and atomic publication.
- `run.py`, `partition.py`: bounded scheduling and disjoint static ownership.

## Provenance

Extracted from the user's R9 `robot_retargeter` working tree, derived from [ccrpRepo/robot_retargeter](https://github.com/ccrpRepo/robot_retargeter). The certified original producer is `09909a3ea5d660da4a35348dbe1803d03b3aeec2a6d268ac617c3f0aa26342eb`; `validation/r9_origin.json` records its files and 45-motion regression membership. Model/mesh provenance is retained from that source tree; no new license grant over upstream code or robot assets is implied. Bones data are obtained separately from [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) and are not redistributed here.
