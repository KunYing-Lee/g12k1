import numpy as np
import scipy.sparse as sp
from scripts.trajectory import equality_tangent_gradient


def test_fixed_boundary_force_is_not_a_free_stationarity_error():
    # The arbitrary force in a fixed halo can be balanced by its equality
    # multiplier; a true gradient on the free coordinate must remain.
    A=sp.csc_matrix([[1.,0.,0.],[0.,1.,0.]])
    result=equality_tangent_gradient(np.array([3.,-8.,4e-7]),A)
    np.testing.assert_array_equal(result,[0.,0.,4e-7])
    assert equality_tangent_gradient(np.array([3.,-8.,.1]),A)[2]==.1


def test_stationarity_projection_is_invariant_to_equality_row_basis():
    A=np.array([[1.,2.,0.],[0.,1.,3.]])
    g=np.array([3.,-2.,7.])
    result=equality_tangent_gradient(g,sp.csc_matrix(A))
    null=np.cross(A[0],A[1]);expected=null*(null@g)/(null@null)
    np.testing.assert_allclose(result,expected,atol=2e-15)
    np.testing.assert_allclose(A@result,0.,atol=5e-15)
    np.testing.assert_allclose(equality_tangent_gradient(g,sp.csc_matrix(np.array([[2.,1.],[-1.,3.]])@A)),result,atol=2e-15)
