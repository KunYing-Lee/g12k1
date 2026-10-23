"""Create a deterministic, complete, disjoint static ownership plan before launch."""
import argparse
import hashlib
import json
from pathlib import Path


def partition(entries, completed, hosts):
    by_member = {e['member']: e for e in entries}
    if len(by_member) != len(entries) or not hosts or any(w <= 0 for w in hosts.values()):
        raise ValueError('Invalid manifest or host weights')
    for member, record in completed.items():
        if member not in by_member or record['source_sha256'] != by_member[member]['sha256']:
            raise ValueError('Completed artifact is not part of the input manifest')
        if record['status'] != 'accepted' or not record.get('output_sha256'):
            raise ValueError('Only verified accepted artifacts may be excluded')
    buckets = [name for name, weight in hosts.items() for _ in range(weight)]
    shards = {name: [] for name in hosts}
    for entry in entries:
        if entry['member'] in completed:
            continue
        digest = hashlib.sha256(entry['member'].encode()).digest()
        owner = buckets[int.from_bytes(digest[:8], 'big') % len(buckets)]
        shards[owner].append(entry)
    assigned = [e['member'] for shard in shards.values() for e in shard]
    if len(assigned) != len(set(assigned)) or set(assigned) | set(completed) != set(by_member):
        raise RuntimeError('Incomplete or overlapping partition')
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--completed', type=Path, required=True, help='Hash-verified accepted-artifact ledger; active workers must first drain')
    parser.add_argument('--hosts', required=True, help='JSON object of positive integer weights, e.g. {"alpha":1,"g1":4}')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    completed = json.loads(args.completed.read_text())
    hosts = json.loads(args.hosts)
    shards = partition(entries, completed, hosts)
    args.output.mkdir(parents=True, exist_ok=False)
    for host, shard in shards.items():
        (args.output / (host+'.json')).write_text(json.dumps(shard, indent=2)+'\n')
    (args.output/'completed.json').write_text(json.dumps(completed, indent=2)+'\n')
    (args.output/'plan.json').write_text(json.dumps(dict(total=len(entries),completed=len(completed),
        hosts=hosts,counts={h:len(s) for h,s in shards.items()},
        input_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.json')}),indent=2)+'\n')


if __name__ == '__main__':
    main()
