import numpy as np
import mujoco
import mink
from scripts.solver import CartesianPoseTask, WholeBodyGroundLimit

def test_cartesian_jacobian_and_position_orientation_independence():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="link"><freejoint/><geom type="box" size=".1 .2 .3"/></body></worldbody></mujoco>')
    c=mink.Configuration(model)
    q=np.array([.2,.3,.8,1.,.2,.3,.4]); q[3:]/=np.linalg.norm(q[3:]);c.update(q)
    task=CartesianPoseTask('link','body',position_cost=3,orientation_cost=2)
    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.exp(np.array([2.,1.,-.4])),np.array([1.,2.,3.])))
    j=task.compute_jacobian(c); numeric=np.zeros_like(j)
    for k in range(model.nv):
        v=np.eye(model.nv)[k]*1e-6
        c.update(q);c.integrate_inplace(v,1);plus=task.compute_error(c)
        c.update(q);c.integrate_inplace(-v,1);minus=task.compute_error(c)
        numeric[:,k]=(plus-minus)/2e-6
    np.testing.assert_allclose(j,numeric,atol=1e-8)
    c.update(q); before=task.compute_error(c)[:3]
    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.exp(np.array([-2.,-1.,.4])),np.array([1.,2.,3.])))
    np.testing.assert_allclose(before,task.compute_error(c)[:3],atol=1e-12)

def test_whole_body_ground_support_jacobian():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="link" pos="0 0 .25"><freejoint/><geom type="box" size=".1 .2 .3"/></body></worldbody></mujoco>')
    c=mink.Configuration(model);ground=WholeBodyGroundLimit(model);q=c.q.copy()
    constraint=ground.compute_qp_inequalities(c,1)
    # All four bottom corners are needed at the flat-face degeneracy.
    assert constraint.G.shape==(4,6)
    assert np.isclose(ground.gaps(c)[0],-.05)
    step=np.array([0,0,.1,.002,-.003,.004]);c.integrate_inplace(step,1)
    predicted=constraint.h-constraint.G@step
    actual=ground.points(c.data,0)[:,2].min()
    assert abs(predicted.min()-actual)<1e-5
