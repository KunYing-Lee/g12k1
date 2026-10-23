import numpy as np
from scripts.trajectory import current_point_multipliers


def test_active_constraint_has_a_current_pose_kkt_certificate():
    g=np.array([-2.,3.]);G=np.array([[1.,0.]]);h=np.array([0.]);A=np.array([[0.,1.]])
    residual,dual=current_point_multipliers(g,G,h,A)
    np.testing.assert_allclose(residual,0.,atol=1e-15)
    np.testing.assert_allclose(dual,[2.],atol=1e-15)


def test_inactive_constraint_cannot_supply_free_stationarity_force():
    residual,dual=current_point_multipliers(np.array([-1.]),np.array([[1.]]),np.array([1.]),np.empty((0,1)))
    assert max(abs(residual).max(),abs(dual).max())>.1


def test_wrong_sign_multiplier_cannot_certify_nonstationary_point():
    residual,dual=current_point_multipliers(np.array([1.]),np.array([[1.]]),np.array([0.]),np.empty((0,1)))
    np.testing.assert_array_equal(dual,[0.])
    np.testing.assert_array_equal(residual,[1.])


def test_redundant_active_features_leave_the_certificate_unchanged():
    g=np.array([-2.,3.]);A=np.array([[0.,1.]])
    residual,dual=current_point_multipliers(g,np.array([[1.,0.],[1.,0.]]),np.zeros(2),A)
    np.testing.assert_allclose(residual,0.,atol=1e-15)
    assert np.all(dual>=0.)
    np.testing.assert_allclose(dual.sum(),2.,atol=1e-15)


def test_positive_constraint_rescaling_preserves_certificate():
    g=np.array([-2.,.2]);G=np.array([[1.,0.],[0.,-1.]]);h=np.array([.003,.01]);A=np.empty((0,2))
    residual,dual=current_point_multipliers(g,G,h,A)
    scales=np.array([1e4,1e-3])
    scaled_residual,scaled_dual=current_point_multipliers(g,G*scales[:,None],h*scales,A)
    np.testing.assert_allclose(scaled_residual,residual,atol=1e-14)
    np.testing.assert_allclose(scaled_dual*scales,dual,atol=1e-14)
