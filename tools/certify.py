"""Certify a refactor against complete accepted R9 regression outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from acceptance import validate_record
from batch import producer_fingerprint, sha256
from run import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference-output',type=Path,required=True)
    p.add_argument('--candidate-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    entries=json.loads(a.manifest.read_text())
    origin=json.loads((ROOT/'validation/r9_origin.json').read_text())
    if len(entries)!=45 or {e['member'] for e in entries}!=set(origin['validated_clips']):
        raise ValueError('Regression membership must match the original 45-motion gate')
    producer=producer_fingerprint();comparisons=[]
    for e in entries:
        records=[];arrays=[]
        for root,expected in [(a.reference_output,origin['producer_sha256']),(a.candidate_output,producer)]:
            record=json.loads((root/'status'/Path(e['member']).with_suffix('.json')).read_text())
            validate_record(record)
            if record['source_sha256']!=e['sha256'] or record['producer_sha256']!=expected:
                raise ValueError('Reference/candidate producer or input differs')
            output=root/'motion'/e['member']
            if sha256(output)!=record['output_sha256']:
                raise ValueError('Output checksum changed')
            q=np.loadtxt(output,delimiter=',')
            if len(q)!=record['frames'] or not np.isfinite(q).all():
                raise ValueError('Incomplete/nonfinite output')
            records.append(record);arrays.append(q)
        np.testing.assert_array_equal(*arrays)
        comparisons.append(dict(member=e['member'],source_sha256=e['sha256'],frames=len(arrays[0]),
            max_qpos_difference=0.,reference_output_sha256=records[0]['output_sha256'],
            output_sha256=records[1]['output_sha256']))
    atomic_json(a.output,dict(passed=True,created_at=time.time(),producer_sha256=producer,
        original_producer_sha256=origin['producer_sha256'],validation_count=len(entries),
        validation_frames=sum(c['frames'] for c in comparisons),exact_numeric_parity=True,
        acceptance_sha256=sha256(ROOT/'validation/acceptance.json'),comparisons=comparisons))


if __name__=='__main__':main()
