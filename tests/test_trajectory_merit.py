"""Nonlinear constrained solutions, including curved active boundaries."""
from types import SimpleNamespace
import numpy as np
import mujoco
import mink
from scripts.trajectory import TrajectoryRetarget
from scripts.solver import solve_equality_reduced_qp


class Disk(mink.Limit):
    def __init__(self, copies):
        self.copies = copies

    def compute_qp_inequalities(self, configuration, dt):
        q = configuration.q
        return mink.Constraint(G=np.repeat((2*q)[None, :], self.copies, axis=0),
                               h=np.full(self.copies, 1.-q@q))


def disk_problem(start, target, copies=1):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".1"/>
      </body></worldbody></mujoco>''')
    task = mink.PostureTask(model, cost=1., gain=1., lm_damping=0.)
    r = SimpleNamespace(model=model, configuration=mink.Configuration(model),
        ik_limits=[Disk(copies)], ik_equalities=lambda:[],
        solve_qp=solve_equality_reduced_qp, solver='daqp',
        self_collision_limit=None, contact_targets=[],
        posture_regularization={'joint_map':{'x':('x',1.,0.),'y':('y',1.,0.)}},
        source_joint_positions=np.asarray([target]),source_joint_names=['x','y'],
        time_step=1.,active_tasks=[task])
    r.project_root_trajectory=lambda _:None
    r.update_targets=lambda _:task.set_target(np.asarray(target))
    r.collision_violation=lambda:max(0.,float(r.configuration.q@r.configuration.q)-1.)
    return TrajectoryRetarget(r,np.asarray([start]),velocity_cost=0.)


def test_curved_boundary_converges_to_analytic_constrained_minimum():
    target=np.array([2.,.7]);expected=target/np.linalg.norm(target)
    t=disk_problem([0.,1.],target)
    q,ok=t._solve()
    assert ok
    np.testing.assert_allclose(q[0],expected,atol=2e-6,rtol=0.)
    assert t.history[-1]['violation']<=1e-8


def test_infeasible_initial_point_restores_same_constrained_minimum():
    target=np.array([2.,.7]);expected=target/np.linalg.norm(target)
    t=disk_problem([1.005,.01],target)
    q,ok=t._solve()
    assert ok
    np.testing.assert_allclose(q[0],expected,atol=2e-6,rtol=0.)


def test_duplicate_support_rows_preserve_physical_constraint_and_solution():
    target=np.array([2.,.7]);t=disk_problem([0.,1.],target,copies=4)
    q,ok=t._solve()
    assert ok
    np.testing.assert_allclose(q[0],target/np.linalg.norm(target),atol=2e-6,rtol=0.)


def test_optimizer_step_box_is_absent_from_physical_certificate():
    from scripts.solver import TangentTrustRegionLimit
    t=disk_problem([0.,0.],[2.,.7])
    t.r.ik_limits.append(TangentTrustRegionLimit(t.r.model,.02,.05,.05))
    p=t.linearize()
    assert p.G.shape==(1,2)
    # At the disk center there is no active physical constraint to balance
    # the objective gradient. A finite update box must not supply that force.
    certificate=t._frame_certificate(0,np.r_[t.q[0],p.q])
    assert certificate[0]==1.
    q,ok=t._solve()
    assert ok
    np.testing.assert_allclose(q[0],np.array([2.,.7])/np.linalg.norm([2.,.7]),atol=2e-6,rtol=0.)
