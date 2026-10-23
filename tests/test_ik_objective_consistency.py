"""Regression tests for IK convergence and floating-base equality constraints."""
import mink
import mujoco
import numpy as np
from scripts.solver import RobotRetarget, RootTranslationEquality


def test_zero_weight_orientation_cannot_stop_position_optimization():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="body"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    configuration = mink.Configuration(model)
    task = mink.FrameTask('body', 'body', position_cost=2., orientation_cost=0.)
    task.set_target(mink.SE3.from_rotation_and_translation(
        mink.SO3.from_rpy_radians(0., 0., 2.), np.zeros(3)))
    r = RobotRetarget.__new__(RobotRetarget)
    r.configuration = configuration
    r.active_tasks = [task]
    # An unrequested 2-radian orientation residual must contribute zero.
    assert np.linalg.norm(task.compute_error(configuration)) > 1.
    assert r.error() == 0.


def test_objective_includes_previous_posture_prior():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="hinge"/><geom type="capsule" size=".1 .2"/></body>
    </worldbody></mujoco>''')
    configuration = mink.Configuration(model)
    posture = mink.PostureTask(model, cost=3.)
    posture.set_target(np.array([.4]))
    r = RobotRetarget.__new__(RobotRetarget)
    r.configuration = configuration
    r.active_tasks = [posture]
    r.regularization_tasks = [posture]
    assert np.isclose(r.error(), 1.2)


def test_root_equality_is_satisfied_by_qp_without_projection():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="body"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    configuration = mink.Configuration(model)
    task = mink.FrameTask('body', 'body', position_cost=100., orientation_cost=1.)
    task.set_target(mink.SE3.from_translation(np.array([2., 3., 1.])))
    equality = RootTranslationEquality(model, 0, ['x', 'y'])
    equality.target[:] = [.3, -.2]
    v = mink.solve_ik(configuration, [task], .01, 'daqp', limits=[], constraints=[equality])
    configuration.integrate_inplace(v, .01)
    np.testing.assert_allclose(configuration.data.qpos[:2], equality.target, atol=1e-8)
    assert configuration.data.qpos[2] > .9


def test_initial_pose_is_fit_before_previous_frame_bounds(tmp_path):
    import pickle
    from scripts.solver import FrameTrustRegionLimit

    xml = tmp_path / 'robot.xml'
    xml.write_text('''<mujoco><worldbody>
      <body name="root"><freejoint/><geom type="sphere" size=".1"/>
        <body name="arm" pos=".2 0 0"><joint name="hinge" axis="0 0 1"/>
          <geom type="capsule" size=".03 .1"/>
        </body>
      </body>
    </worldbody></mujoco>''')
    motion = tmp_path / 'motion.pkl'
    qarm = np.array([np.cos(.4), 0., 0., np.sin(.4)])
    with motion.open('wb') as f:
        pickle.dump(dict(keypoint_names=['root', 'arm'],
            positions=np.array([[[1., 0., 1.], [1.2, 0., 1.]]]),
            quaternions=np.array([[[1., 0., 0., 0.], qarm]]),
            contact_names=[], contact_states=np.zeros((1, 0)), fps=120), f)
    r = RobotRetarget(str(xml), str(motion),
        {'root': ['root', 100., 100.], 'arm': ['arm', 0., 10.]},
        temporal_regularization={'enabled': True, 'cost': 2.},
        ik_convergence_tolerance=1e-8)
    bound = FrameTrustRegionLimit(r.model, .01, .01, .01)
    r.frame_trust_region_limit = bound
    r.ik_limits.append(bound)
    r.initialize_first_pose()
    # Initialization is an independent pose fit, not the first 0.01-rad step
    # toward the observation. The observation remains the first output frame.
    assert abs(r.configuration.data.qpos[-1] - .8) < 1e-4
    assert bound in r.ik_limits
    assert r.temporal_posture_task in r.motion_tasks
    assert r.result_pos == []


def test_collision_sqp_restores_clearance_when_objective_must_increase():
    from scripts.solver import RestoringCollisionAvoidanceLimit
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint type="slide" axis="0 1 0"/><geom name="fixed" type="sphere" size=".1"/></body>
      <body pos=".2 0 0"><joint type="slide" axis="1 0 0"/>
        <geom name="moving" type="sphere" size=".1"/>
      </body></worldbody></mujoco>''')
    r = RobotRetarget.__new__(RobotRetarget)
    r.model = model
    r.configuration = mink.Configuration(model)
    r.configuration.update(q=np.array([0., .000995]))
    limit = RestoringCollisionAvoidanceLimit(model,
        geom_pairs=[(['fixed'], ['moving'])], minimum_distance_from_collisions=.001,
        collision_detection_distance=.03, gain=.85, recovery_gain=.5)
    a = limit.compute_qp_inequalities(r.configuration, .001)
    b = limit.compute_qp_inequalities(r.configuration, .02)
    np.testing.assert_allclose(a.h, b.h)
    # Also check the safe side, where the old implementation mixed units.
    r.configuration.update(q=np.array([0., .002]))
    a = limit.compute_qp_inequalities(r.configuration, .001)
    b = limit.compute_qp_inequalities(r.configuration, .02)
    np.testing.assert_allclose(a.h, b.h)
    assert np.isclose(a.h[0], .85 * .001)
    r.configuration.update(q=np.array([0., .000995]))
    task = mink.PostureTask(model, cost=100.)
    task.set_target(np.zeros(2))
    r.active_tasks = [task]
    r.ik_limits = [limit]
    r.ik_equalities = lambda: []
    r.self_collision_limit = limit
    r.self_collision_minimum_distance = .001
    r.self_collision_tolerance = 1e-5
    r.self_collision_max_backtracks = 12
    r.max_iter, r.damping, r.solver = 50, 1., 'daqp'
    r.ik_convergence_tolerance = 1e-10
    r.solver_diagnostics = []
    initial = r.error()
    r.solve_frame(0)
    assert r.error() > initial  # Necessary, correct constrained ascent.
    assert r.collision_violation() < 1e-8
    assert r.solver_diagnostics[-1]['reason'] != 'no_descent'


def test_trust_region_resolves_curved_constraint_without_scaled_restoration():
    # Feasible set y >= x^2 + margin: a tangent move at the boundary
    # needs a second-order normal correction, which blind scaling misses.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    class ParabolicClearance(mink.Limit):
        recovery_gain = .5
        def compute_qp_inequalities(self, configuration, dt):
            x, y = configuration.data.qpos
            gap = y - x*x - .001
            return mink.Constraint(G=np.array([[2*x, -1.]]),
                h=np.array([(.85 if gap >= 0 else .5) * gap]))
    r = RobotRetarget.__new__(RobotRetarget)
    r.model, r.configuration = model, mink.Configuration(model)
    r.configuration.update(q=np.array([0., .001]))
    task = mink.PostureTask(model, cost=10.)
    task.set_target(np.array([.05, 0.]))
    r.active_tasks, r.ik_limits = [task], [ParabolicClearance()]
    r.ik_equalities = lambda: []
    r.self_collision_limit = r.ik_limits[0]
    r.minimum_self_collision_distance = lambda: float(r.configuration.data.qpos[1] - r.configuration.data.qpos[0]**2)
    r.collision_violation = lambda: max(0., .001-r.minimum_self_collision_distance())
    r.self_collision_minimum_distance, r.self_collision_tolerance = .001, 1e-5
    r.max_iter, r.damping, r.solver = 200, 1., 'daqp'
    r.ik_convergence_tolerance, r.solver_diagnostics = 1e-8, []
    r.solve_frame(0)
    x, y = r.configuration.data.qpos
    assert abs(x - .05 + 2*x*y) < 1e-4  # Constrained first-order optimality.
    assert abs(y - x*x - .001) < 1e-8
    assert r.solver_diagnostics[-1]['reason'] in ('kkt_tolerance',)
