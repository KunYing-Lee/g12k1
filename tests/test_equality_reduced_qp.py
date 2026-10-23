from pathlib import Path
import numpy as np
import qpsolvers
from scripts.solver import solve_equality_reduced_qp


def assert_kkt(p, s, tol):
    assert s.found
    assert np.max(p.G @ s.x-p.h) <= tol
    np.testing.assert_allclose(p.A @ s.x, p.b, atol=tol)
    np.testing.assert_allclose(p.P @ s.x+p.q+p.G.T @ s.z+p.A.T @ s.y, 0, atol=tol)
    finite=np.isfinite(p.h)
    assert np.max(abs(s.z[finite]*(p.G @ s.x-p.h)[finite])) <= tol
    assert np.min(s.z) >= -tol


def test_nonzero_equality_shift_and_original_multipliers():
    p=qpsolvers.Problem(P=np.array([[3.,1.],[1.,2.]]),q=np.array([-8.,-1.]),
        G=np.array([[1.,0.],[-1.,0.]]),h=np.array([.5,2.]),
        A=np.array([[1.,1.]]),b=np.array([1.]))
    s=solve_equality_reduced_qp(p,solver='daqp',primal_tol=1e-12)
    np.testing.assert_allclose(s.x,[.5,.5],atol=1e-12)
    assert_kkt(p,s,1e-10)


def test_real_warmup_834_qp_is_feasible_without_relaxation():
    # Frozen QP from Bones warm_up_legs_R_002__A359_M, native frame 834.
    # The unreduced DAQP formulation reports failure despite x=0 feasible.
    z=np.load(Path(__file__).parent/'fixtures/warmup_834_qp.npz')
    p=qpsolvers.Problem(**{k:z[k] for k in z.files})
    assert np.min(p.h)>=0 and np.max(abs(p.b))==0
    s=solve_equality_reduced_qp(p,solver='daqp',primal_tol=1e-12)
    assert_kkt(p,s,1e-10)


def test_inconsistent_equalities_are_not_silently_relaxed():
    p=qpsolvers.Problem(P=np.eye(2),q=np.zeros(2),A=np.array([[1.,0.],[1.,0.]]),b=np.array([0.,1.]))
    assert not solve_equality_reduced_qp(p,solver='daqp',primal_tol=1e-12).found
