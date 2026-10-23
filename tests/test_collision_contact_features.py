"""Multiple closest witnesses must constrain every closing direction."""
import mink
import mujoco
import numpy as np
from scripts.solver import RestoringCollisionAvoidanceLimit


def test_mesh_edge_over_cylinder_cap_has_two_directional_constraints():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="wedge" vertex="-.1 0 0 .1 0 0 -.1 .02 .02 .1 .02 .02
                                  -.1 -.02 .02 .1 -.02 .02"/>
      </asset><worldbody>
      <body name="disk"><freejoint/><geom name="cylinder" type="cylinder" size=".04 .03"/></body>
      <body name="edge" pos="0 0 .031"><freejoint/><geom name="mesh" type="mesh" mesh="wedge"/></body>
      </worldbody></mujoco>''')
    config = mink.Configuration(model)
    limit = RestoringCollisionAvoidanceLimit(model,
        geom_pairs=[(['mesh'], ['cylinder'])], minimum_distance_from_collisions=.001,
        collision_detection_distance=.03)
    mesh, cylinder = model.geom('mesh').id, model.geom('cylinder').id
    rows, bounds = limit.cap_feature_constraints(config, mesh, cylinder)
    G, h = np.asarray(rows), np.asarray(bounds)
    assert len(h) >= 2
    # Higher edge/cap intersections are not closest-distance derivatives.
    assert np.max(h) <= 1e-7
    # Only rotate the wedge around y. Opposite ends close in opposite directions.
    direction = np.zeros(model.nv)
    direction[model.jnt_dofadr[1]+4] = 1.
    active = np.abs(h) < 1e-8
    assert (G[active] @ direction).max() > .039
    assert (G[active] @ -direction).max() > .039
    # The maximum over witness derivatives agrees with the one-sided true
    # distance derivative; a single GJK witness cannot encode both sides.
    q0 = config.data.qpos.copy()
    d0 = mujoco.mj_geomDistance(model, config.data, mesh, cylinder, .03, None)
    for sign in [-1., 1.]:
        config.update(q=q0)
        config.integrate_inplace(sign*direction, 1e-5)
        d1 = mujoco.mj_geomDistance(model, config.data, mesh, cylinder, .03, None)
        assert np.isclose((d0-d1)/1e-5, (G[active] @ (sign*direction)).max(), atol=2e-5)


def test_box_face_under_mesh_face_retains_both_edge_orders():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="plate" vertex="-.1 -.1 0 .1 -.1 0 .1 .1 0 -.1 .1 0
                                  -.1 -.1 .02 .1 -.1 .02 .1 .1 .02 -.1 .1 .02"/>
      </asset><worldbody>
      <body><freejoint/><geom name="box" type="box" size=".04 .04 .03"/></body>
      <body pos="0 0 .031"><freejoint/><geom name="mesh" type="mesh" mesh="plate"/></body>
      </worldbody></mujoco>''')
    config = mink.Configuration(model)
    limit = RestoringCollisionAvoidanceLimit(model,
        geom_pairs=[(['mesh'], ['box'])],minimum_distance_from_collisions=.001,
        collision_detection_distance=.03)
    mesh,box=model.geom('mesh').id,model.geom('box').id
    rows,bounds=limit.polytope_feature_constraints(config,mesh,box)
    rows,bounds=np.asarray(rows),np.asarray(bounds)
    assert len(rows)>=4
    q=config.q.copy();distance=mujoco.mj_geomDistance(model,config.data,mesh,box,.03,None)
    for sign in (-1.,1.):
        direction=np.zeros(model.nv);direction[9]=sign
        predicted=np.max(rows@direction-bounds)
        config.update(q);config.integrate_inplace(direction,1e-6)
        actual=-(mujoco.mj_geomDistance(model,config.data,mesh,box,.03,None)-distance)/1e-6
        np.testing.assert_allclose(predicted,actual,atol=2e-5)
    config.update(q)


def test_cylinder_barrel_face_contact_keeps_generator_endpoints():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom name="cylinder" type="cylinder" size=".036 .047"/></body>
      <body pos=".047 0 0"><freejoint/><geom name="face" type="box" size=".01 .1 .1"/></body>
      </worldbody></mujoco>''')
    model.opt.ccd_tolerance=1e-12
    model.opt.ccd_iterations=100
    config=mink.Configuration(model)
    limit=RestoringCollisionAvoidanceLimit(model,geom_pairs=[(['face'],['cylinder'])],
        minimum_distance_from_collisions=.001,collision_detection_distance=.03)
    face,cylinder=model.geom('face').id,model.geom('cylinder').id
    rows,bounds=limit.barrel_feature_constraints(config,face,cylinder)
    rows,bounds=np.asarray(rows),np.asarray(bounds);assert len(rows)>=2
    q=config.q.copy();before=mujoco.mj_geomDistance(model,config.data,face,cylinder,.03,None)
    for sign in (-1.,1.):
        delta=np.zeros(model.nv);delta[10]=sign
        predicted=np.max(rows@delta-bounds)
        config.update(q);config.integrate_inplace(delta,1e-6)
        actual=-(mujoco.mj_geomDistance(model,config.data,face,cylinder,.03,None)-before)/1e-6
        np.testing.assert_allclose(predicted,actual,atol=2e-5)


def test_single_pass_distance_rows_match_original_restoring_constraints():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom name="a" type="sphere" size=".04"/></body>
      <body pos=".08 0 0"><freejoint/><geom name="b" type="sphere" size=".04"/></body>
      </worldbody></mujoco>''')
    config = mink.Configuration(model)
    limit = RestoringCollisionAvoidanceLimit(model, geom_pairs=[(['a'],['b'])],
        minimum_distance_from_collisions=.001, collision_detection_distance=.03,
        gain=.85, recovery_gain=.5)
    for x in [.07,.08,.0805,.081,.09,.2]:
        pose=config.q.copy();pose[7]=x;config.update(pose)
        old=mink.CollisionAvoidanceLimit.compute_qp_inequalities(limit,config,1.)
        h=old.h.copy()
        distance=mujoco.mj_geomDistance(model,config.data,0,1,.001,None)
        if distance<.001:h[0]=min(h[0],-.5*(.001-distance))
        new=limit.compute_qp_inequalities(config,1.)
        np.testing.assert_allclose(new.G,old.G,atol=0.,rtol=0.)
        np.testing.assert_allclose(new.h,h,atol=1e-16,rtol=0.)
