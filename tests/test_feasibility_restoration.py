"""Infeasible trust boxes require a normal step, not relaxed final constraints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import qpsolvers
from solver import feasibility_normal_step


def test_normal_step_reduces_violation_without_breaking_root_equality():
    # Ground requires z >= .12, but an optimizer step is at most .02.
    p=qpsolvers.Problem(P=np.eye(2),q=np.zeros(2),
        G=np.array([[0.,-1.],[1.,0.],[-1.,0.]]),h=np.array([-.12,.1,.1]),
        A=np.array([[1.,0.]]),b=np.zeros(1))
    d,reduction=feasibility_normal_step(p,.02)
    np.testing.assert_allclose(d,[0.,.02],atol=1e-12)
    assert abs(reduction-.02)<1e-12
    assert np.max(p.G@d-p.h)>.09  # A normal step is not an acceptance certificate.
    np.testing.assert_array_equal(p.h,[-.12,.1,.1])


def test_restoration_model_reaches_original_feasible_set():
    height=0.
    for _ in range(6):
        p=qpsolvers.Problem(P=np.eye(1),q=np.zeros(1),G=np.array([[-1.]]),h=np.array([height-.12]))
        d,_=feasibility_normal_step(p,.02)
        height+=d[0]
    assert abs(height-.12)<1e-12


def test_normal_step_keeps_joint_limits_hard():
    p=qpsolvers.Problem(P=np.eye(1),q=np.zeros(1),
        G=np.array([[-1.],[1.]]),h=np.array([-.12,.01]))
    d,reduction=feasibility_normal_step(p,.02,soft_rows=[True,False])
    assert abs(d[0]-.01)<1e-12
    assert abs(reduction-.01)<1e-12


def test_captured_motion_initialization_has_a_restoring_normal_step():
    from scipy.optimize import linprog
    p=np.load(Path(__file__).parent/'fixtures/initial_ground_trust_qp.npz')
    n=len(p['q']);G,h=p['G'],p['h'];finite=np.isfinite(h)
    nominal=qpsolvers.Problem(P=p['P'],q=p['q'],G=G[:-2*n],h=h[:-2*n],A=p['A'],b=p['b'])
    strict=linprog(np.zeros(n),A_ub=G[finite],b_ub=h[finite],A_eq=p['A'],b_eq=p['b'],bounds=[(None,None)]*n)
    assert strict.status==2  # Captured initial ground penetration exceeds the trust box.
    # Fixture's scalar rows are hard joint/trust bounds; geometric rows have
    # multiple coefficients. Production selects rows by their limit object.
    soft=np.count_nonzero(nominal.G,axis=1)>1
    delta,predicted=feasibility_normal_step(nominal,.05,soft_rows=soft)
    assert predicted>0
    hard=(~soft)&np.isfinite(nominal.h)
    assert np.max(nominal.G[hard]@delta-nominal.h[hard])<=1e-10
    assert np.max(abs(p['A']@delta-p['b']))<=1e-12
