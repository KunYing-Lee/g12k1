import copy
import pytest
from scripts.acceptance import validate_record
from scripts.partition import partition


def accepted():
    return dict(status='accepted',trajectory_converged=True,frames=4,expected_frames=4,fps=120,
        final_kkt=dict(stationarity=1e-7,complementarity=1e-7,violation=0.,
                       stationarity_normalization='per_frame',complementarity_normalization='absolute'),
        minimum_ground_gap_m=0.,minimum_self_clearance_m=.001,
        joint_limit_violation_rad=0.,root_xy_error_m=0.)


@pytest.mark.parametrize('field,value',[('stationarity',1.01e-6),('complementarity',1.01e-6),('violation',1.01e-8),('stationarity',float('nan'))])
def test_rejects_solver_certificate_outside_original_gate(field,value):
    record=accepted();record['final_kkt'][field]=value
    with pytest.raises(ValueError):validate_record(record)


def test_static_partition_is_disjoint_complete_and_reproducible():
    entries=[dict(member=f'g1/{i}.csv',sha256=str(i)) for i in range(100)]
    completed={'g1/3.csv':dict(status='accepted',source_sha256='3',output_sha256='saved')}
    shards=partition(entries,completed,{'alpha':1,'g1':4})
    assert shards==partition(entries,completed,{'alpha':1,'g1':4})
    a,b=({e['member'] for e in shards[k]} for k in ['alpha','g1'])
    assert not a&b
    assert a|b|set(completed)=={e['member'] for e in entries}
    assert len(a)+len(b)==99


def test_partition_rejects_stale_completed_source():
    with pytest.raises(ValueError):partition([dict(member='x',sha256='new')],{'x':dict(status='accepted',source_sha256='old',output_sha256='saved')},{'g1':1})
