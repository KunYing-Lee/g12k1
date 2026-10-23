"""Bounded complete-motion workers; independent processes and atomic status files."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

from batch import run_clip, producer_fingerprint, sha256
from acceptance import validate_record


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def check_manifest(entries):
    members = [e['member'] for e in entries]
    if len(members) != len(set(members)):
        raise ValueError('Duplicate motion assignment')
    for entry in entries:
        member = Path(entry['member'])
        if member.is_absolute() or '..' in member.parts:
            raise ValueError('Unsafe motion member')
        if not Path(entry['path']).is_absolute():
            raise ValueError('Manifest source paths must be absolute')


def accepted_record(entry, output, producer):
    record = output / 'status' / Path(entry['member']).with_suffix('.json')
    destination = output / 'motion' / entry['member']
    if not record.exists():
        return None
    old = json.loads(record.read_text())
    if old.get('status') != 'accepted':
        return None
    validate_record(old)
    if (old['producer_sha256'] != producer or old['source_sha256'] != entry['sha256']
            or not destination.exists() or sha256(destination) != old['output_sha256']):
        raise ValueError(f'Accepted artifact identity mismatch: {entry["member"]}')
    return old


def run(args):
    entries = [json.loads(args.entry_json)] if args.entry_json else json.loads(args.manifest.read_text())
    check_manifest(entries)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'work').mkdir(exist_ok=True)
    if args.entry_json is not None:
        result = run_clip(entries[0], output)
        if result['status'] == 'accepted':
            validate_record(result)
            return 0
        return 1
    if args.workers < 1 or args.frame_budget < 1:
        raise ValueError('Worker and frame budgets must be positive')
    lock = (output / '.manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    producer = producer_fingerprint()
    if not args.validation:
        if args.gate is None:
            raise ValueError('Production requires a matching 45-motion validation certificate')
        gate = json.loads(args.gate.read_text())
        if (gate.get('passed') is not True or gate.get('validation_count') != 45
                or gate.get('producer_sha256') != producer):
            raise ValueError('Validation certificate does not match this producer')
    frame_workers = int(os.environ.get('RETARGET_FRAME_WORKERS', '2'))
    if frame_workers < 1:
        raise ValueError('Frame workers must be positive')
    identity = dict(producer_sha256=producer, input_manifest_sha256=sha256(args.manifest),
                    workers=args.workers, frame_workers=frame_workers,
                    max_active_frames=args.frame_budget, oversized_clip_frame_workers=6,
                    validation=args.validation)
    config = output / 'batch_config.json'
    if config.exists():
        old = json.loads(config.read_text())
        for key in ('producer_sha256', 'input_manifest_sha256', 'validation'):
            if old[key] != identity[key]:
                raise ValueError(f'Output root has a different {key}')
    atomic_json(config, identity)
    active = {}
    index = 0
    stats = dict(total=len(entries), completed=0, accepted=0, failed=0,
                 resumed=0, started_at=time.time())
    draining = False

    def drain(signum, frame):
        nonlocal draining
        draining = True

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)
    (output / 'worker_logs').mkdir(exist_ok=True)

    def finish(result, resumed=False):
        stats['completed'] += 1
        stats[result['status']] += 1
        stats['resumed'] += int(resumed)
        with (output / 'events.jsonl').open('a') as events:
            events.write(json.dumps({**result, 'resumed': resumed}) + '\n')

    while index < len(entries) or active:
        while index < len(entries) and not draining:
            entry = entries[index]
            old = accepted_record(entry, output, producer)
            if old is not None:
                finish(old, True)
                index += 1
                continue
            frames = entry.get('frames')
            if frames is None:
                with open(entry['path']) as stream:
                    frames = sum(bool(line.strip()) for line in stream) - 1
                entry['frames'] = frames
            if len(active) >= args.workers or (active and sum(j['frames'] for j in active.values()) + frames > args.frame_budget):
                break
            fw = 6 if frames > args.frame_budget else frame_workers
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', RETARGET_FRAME_WORKERS=str(fw))
            command = [sys.executable, str(Path(__file__).resolve()), '--entry-json', json.dumps(entry),
                       '--output', str(output)]
            with (output / 'worker_logs' / f'{index:06d}.log').open('w') as log:
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active[child.pid] = dict(process=child, entry=entry, frames=frames, frame_workers=fw)
            index += 1
        for pid in list(active):
            job = active[pid]
            if job['process'].poll() is None:
                continue
            del active[pid]
            entry = job['entry']
            record = output / 'status' / Path(entry['member']).with_suffix('.json')
            if not record.exists():
                result = dict(member=entry['member'], source_sha256=entry['sha256'], producer_sha256=producer,
                              status='failed', error=f'Worker exited {job["process"].returncode} without a result')
            else:
                result = json.loads(record.read_text())
            if result['status'] == 'accepted':
                if job['process'].returncode != 0 or result['frames'] != job['frames']:
                    raise RuntimeError('Accepted worker exit/frame-count mismatch')
                accepted_record(entry, output, producer)
            result['execution_frame_workers'] = job['frame_workers']
            atomic_json(record, result)
            finish(result)
        stats.update(elapsed_seconds=time.time()-stats['started_at'], active_clips=len(active),
                     active_frames=sum(j['frames'] for j in active.values()), draining=draining)
        atomic_json(output / 'summary.json', stats)
        if draining and not active:
            break
        if active:
            time.sleep(1)
    stats['finished_at'] = time.time()
    atomic_json(output / 'summary.json', stats)
    return int(stats['failed'] != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--frame-budget', type=int, default=12000)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--validation', action='store_true', help='Run acceptance validation, never relax the motion gate')
    parser.add_argument('--entry-json', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.manifest is None) == (args.entry_json is None):
        parser.error('Supply exactly one manifest or internal worker entry')
    sys.exit(run(args))


if __name__ == '__main__':
    main()
