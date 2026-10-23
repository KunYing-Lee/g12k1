"""Conditional temporal solves retain every boundary velocity residual."""
import numpy as np
import mujoco,mink
from types import SimpleNamespace
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scripts.solver import RootTranslationEquality,solve_equality_reduced_qp
from scripts.trajectory import TrajectoryRetarget,_solve_block,velocity_operator


def make_problem():
    m=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body><freejoint/><geom type="sphere" size=".1"/>
      <body><joint name="j" limited="true" range="-2 2"/><geom type="sphere" size=".1"/>
      </body></body></worldbody></mujoco>''')
    source=np.sin(np.arange(20)*.4)*.5
    task=mink.PostureTask(m,cost=1.,gain=1.,lm_damping=0.)
    eq=RootTranslationEquality(m,0,['x','y'])
    r=SimpleNamespace(model=m,configuration=mink.Configuration(m),
        self_collision_limit=None,ik_limits=[mink.ConfigurationLimit(m)],contact_targets=[],
        posture_regularization={'joint_map':{'j':('source',1.,0.)}},
        source_joint_positions=source[:,None],source_joint_names=['source'],time_step=1.,
        active_tasks=[task],ik_equalities=lambda:[eq],collision_violation=lambda:0.,
        solve_qp=solve_equality_reduced_qp,solver='daqp')
    r.project_root_trajectory=lambda _:None
    def update(i):
        q=m.qpos0.copy();q[-1]=.8*source[i];task.set_target(q)
    r.update_targets=update
    q=np.repeat(m.qpos0[None,:],20,axis=0)
    t=TrajectoryRetarget(r,q,velocity_cost=.5)
    return t,source


def test_block_objective_change_matches_complete_objective_including_halos():
    t,_=make_problem();before=t.q.copy()
    start,end,left,right=5,11,4,12
    a,b,poses,report=_solve_block(t,(start,end,left,right,before[left:right],100,1e-8))
    assert report['converged']
    after=before.copy();after[a:b]=poses
    delta=t.objective_difference(before,after)
    np.testing.assert_allclose(delta,report['objective_after']-report['objective_before'],atol=1e-12)
    np.testing.assert_array_equal(after[:start],before[:start])
    np.testing.assert_array_equal(after[end:],before[end:])
    assert delta<0.


def test_overlapping_solve_then_global_certificate_matches_undecomposed_optimum():
    t,source=make_problem();t.schwarz_sweep(6,100,1e-8)
    result,ok=t._solve(100,1e-8);assert ok
    V=velocity_operator(20,1,np.array([0]),1.,.5)
    exact=spsolve(eye(20,format='csc')+V.T@V,.8*source+V.T@(V@source))
    np.testing.assert_allclose(result[:,-1],exact,atol=1e-7)


def test_parallel_schwarz_matches_serial_conditional_solves():
    import sys,pytest
    if sys.platform != 'linux':
        pytest.skip('Fork frame workers are supported on Linux')
    serial,_=make_problem();parallel,_=make_problem()
    serial.frame_workers=1;parallel.frame_workers=2
    try:
        serial.schwarz_sweep(6,100,1e-8)
        parallel.schwarz_sweep(6,100,1e-8)
        np.testing.assert_allclose(parallel.q,serial.q,atol=1e-12,rtol=0.)
        a,ok_a=serial._solve(100,1e-8);b,ok_b=parallel._solve(100,1e-8)
        assert ok_a and ok_b
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0.)
    finally:
        if parallel._pool is not None:parallel._pool.shutdown()
