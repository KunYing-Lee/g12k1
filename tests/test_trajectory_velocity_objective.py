import sys
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trajectory import velocity_operator, kkt_certificate


def test_source_velocity_tracking_preserves_fast_motion():
    """Unlike Dq regularization, D(q-source) does not attenuate source motion."""
    source = np.sin(np.arange(100) * 2.1)
    v = velocity_operator(100, 1, np.array([0]), 1 / 120, .1)
    from scipy.sparse import eye
    solution = spsolve(eye(100, format='csc') + v.T @ v,
                      source + v.T @ (v @ source))
    np.testing.assert_allclose(solution, source, atol=1e-12)


def test_velocity_jacobian_and_cross_frame_hessian():
    rng = np.random.default_rng(27)
    q, source, delta = [rng.normal(size=(7, 9)) for _ in range(3)]
    dofs = np.array([2, 5, 8])
    v = velocity_operator(7, 9, dofs, .01, .1)
    residual = lambda x: (np.diff((x-source)[:, dofs], axis=0)*10).ravel()
    np.testing.assert_allclose((residual(q+1e-6*delta)-residual(q))/1e-6,
                               v @ delta.ravel(), atol=1e-7)
    assert (v.T @ v)[2, 11] < 0






def test_unrelated_large_gradient_cannot_hide_a_nonconverged_frame():
    residual = np.array([2e-6, 0., 0., 0.])
    objective_gradient = np.array([1., 0., 1e9, 0.])
    errors, comp = kkt_certificate(residual, objective_gradient,
                                  np.array([1.]), np.array([.01]), 2)
    assert errors[0] == 2e-6
    assert errors.max() > 1e-6
    assert comp == .01




def test_direct_tangent_integration_matches_configuration_integration():
    import mujoco,mink
    from types import SimpleNamespace
    from trajectory import TrajectoryRetarget
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <freejoint/><geom type="sphere" size=".1"/><body pos="0 0 .2">
      <joint axis="0 1 0"/><geom type="sphere" size=".1"/>
      </body></body></worldbody></mujoco>''')
    rng=np.random.default_rng(19);q=np.repeat(model.qpos0[None,:],5,axis=0)
    for pose in q:mujoco.mj_integratePos(model,pose,rng.normal(size=model.nv),.3)
    delta=rng.normal(size=(5,model.nv))*.05
    t=TrajectoryRetarget.__new__(TrajectoryRetarget);t.r=SimpleNamespace(model=model);t.q=q;t.fixed_frames=set()
    result=t.integrate(delta.ravel());config=mink.Configuration(model)
    for pose,dq,actual in zip(q,delta,result):
        config.update(pose);config.integrate_inplace(dq,1.)
        np.testing.assert_array_equal(actual,config.q)
