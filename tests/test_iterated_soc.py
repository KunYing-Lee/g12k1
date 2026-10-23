import numpy as np
import mujoco,mink
from types import SimpleNamespace
from scripts.solver import solve_equality_reduced_qp
from scripts.trajectory import TrajectoryRetarget


class CurvedLimit(mink.Limit):
    def compute_qp_inequalities(self, configuration, dt):
        x=configuration.q[0]
        return mink.Constraint(G=np.array([[2*x]]),h=np.array([1-x*x]))


def trajectory():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0"/><geom type="sphere" size=".1"/>
      </body></worldbody></mujoco>''')
    r=SimpleNamespace(model=model,configuration=mink.Configuration(model),
        ik_limits=[CurvedLimit()],ik_equalities=lambda:[],solve_qp=solve_equality_reduced_qp,solver='daqp')
    r.collision_violation=lambda:max(0.,r.configuration.q[0]**2-1.)
    t=TrajectoryRetarget.__new__(TrajectoryRetarget);t.r=r;t.frames=[0];t.fixed_frames=set()
    t.select=lambda i,q:r.configuration.update(q)
    return t


def test_iterated_soc_reduces_curved_constraint_defect_before_acceptance():
    t=trajectory();q=np.array([1.1]);corrected=t._frame_correct(0,q,.2)
    assert corrected is not None
    # A single Newton correction leaves ~.0091 violation; the iterated
    # correction meets the same 1e-8 nonlinear feasibility gate.
    assert max(0.,corrected[0]**2-1.)<=1e-8
    assert abs(corrected[0]-q[0])<=.2
    np.testing.assert_array_equal(q,[1.1])


def test_soc_does_not_escape_its_total_displacement_budget():
    assert trajectory()._frame_correct(0,np.array([1.1]),.05) is None
