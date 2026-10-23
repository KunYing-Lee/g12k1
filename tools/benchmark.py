"""Two matched rounds of 16x2/32x2/48x2 on identical complete source motions."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from acceptance import validate_record
from batch import producer_fingerprint, sha256
from run import atomic_json


def memory_sample():
    cg = Path('/sys/fs/cgroup')
    return {name: (cg/name).read_text() for name in ('cpu.stat','cpu.pressure','memory.current','memory.stat','memory.events','memory.pressure')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', type=Path, required=True)
    args = p.parse_args()
    entries = json.loads(args.manifest.read_text())
    if len(entries) < 96:
        raise ValueError('At least two waves for 48 workers are required')
    args.output.mkdir(parents=True, exist_ok=False)
    gate = json.loads(args.gate.read_text())
    producer = producer_fingerprint()
    if not gate['passed'] or gate['producer_sha256'] != producer:
        raise ValueError('Benchmark requires the validated release')
    results = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', RETARGET_FRAME_WORKERS='2')
    reference = None
    for round_id, workers in enumerate((16,32,48,48,32,16)):
        out = args.output/f'run_{round_id}_{workers}x2'
        command = [sys.executable, str(ROOT/'scripts/run.py'), '--manifest', str(args.manifest.resolve()),
                   '--output', str(out), '--workers', str(workers), '--frame-budget', '72000', '--gate', str(args.gate.resolve())]
        start = time.monotonic()
        with (args.output/f'run_{round_id}.log').open('w') as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        samples = []
        while child.poll() is None:
            samples.append(dict(elapsed=time.monotonic()-start, **memory_sample()))
            summary = out/'summary.json'
            progress = json.loads(summary.read_text()) if summary.exists() else {}
            atomic_json(args.output/'progress.json',dict(round=round_id,workers=workers,total_rounds=6,**progress))
            time.sleep(5)
        elapsed = time.monotonic()-start
        atomic_json(out/'resource_samples.json',samples)
        if child.returncode != 0:
            raise RuntimeError(f'Benchmark run {round_id} failed; inspect {out}')
        max_diff = 0.
        frames = 0
        for entry in entries:
            record = json.loads((out/'status'/Path(entry['member']).with_suffix('.json')).read_text())
            validate_record(record)
            if record['producer_sha256'] != producer or record['source_sha256'] != entry['sha256']:
                raise ValueError('Producer/source identity changed')
            path = out/'motion'/entry['member']
            if sha256(path) != record['output_sha256']:
                raise ValueError('Published output checksum changed')
            q = np.loadtxt(path,delimiter=',');frames += len(q)
            if reference is not None:
                ref = np.loadtxt(reference/'motion'/entry['member'],delimiter=',')
                np.testing.assert_array_equal(q,ref)
                max_diff = max(max_diff,float(abs(q-ref).max()))
        if reference is None:
            reference = out
        results.append(dict(round=round_id,workers=workers,frame_workers=2,clips=len(entries),frames=frames,
                            seconds=elapsed,clips_per_hour=len(entries)*3600/elapsed,
                            max_qpos_difference=max_diff,output=str(out)))
        atomic_json(args.output/'results.json',results)
    medians = {str(w):statistics.median(r['seconds'] for r in results if r['workers']==w) for w in (16,32,48)}
    winner = min(medians,key=medians.get)
    atomic_json(args.output/'selection.json',dict(workers=int(winner),frame_workers=2,
        median_seconds=medians,clips_per_hour=len(entries)*3600/medians[winner],
        producer_sha256=producer,input_manifest_sha256=sha256(args.manifest),
        accepted_runs=len(results)*len(entries),max_qpos_difference=0.))


if __name__ == '__main__':
    main()
