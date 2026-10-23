import numpy as np
import scipy.sparse as sp
import qpsolvers
from scripts.trajectory import consolidate_coordinate_bounds


def test_exact_bound_intersection_and_original_dual_stationarity():
    p=qpsolvers.Problem(P=sp.eye(2,format='csc'),q=np.array([-3.,-4.]),
        G=sp.csc_matrix([[1.,0.],[2.,0.],[-1.,0.],[0.,1.],[0.,3.],[1.,1.]]),
        h=np.array([2.,2.,2.,3.,6.,2.5]))
    reduced,selected,scales=consolidate_coordinate_bounds(p)
    assert reduced.G.shape[0]==4
    old=qpsolvers.solve_problem(p,solver='clarabel',tol_gap_abs=1e-12,tol_feas=1e-12)
    new=qpsolvers.solve_problem(reduced,solver='clarabel',tol_gap_abs=1e-12,tol_feas=1e-12)
    np.testing.assert_allclose(new.x,old.x,atol=1e-9)
    dual=np.zeros(len(p.h));dual[selected]=new.z/scales
    assert np.max(abs(p.P@new.x+p.q+p.G.T@dual))<1e-10
    assert np.max(abs(dual*(p.h-p.G@new.x)))<1e-9
    rng=np.random.default_rng(1)
    for x in rng.uniform(-4,4,(500,2)):
        assert np.all(p.G@x<=p.h)==np.all(reduced.G@x<=reduced.h)
