from pathlib import Path
import json
import mink,mujoco
import numpy as np
import pytest
from scripts.solver import RestoringCollisionAvoidanceLimit, RobotRetarget


@pytest.mark.parametrize('position,expected',[
    ('2 0 0',.9), ('0 0 2',.9), ('2 0 2',np.sqrt(2.)*.9)])
def test_face_cap_and_rim_distances_have_analytic_values(position,expected):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body><freejoint/><geom name="cylinder" type="cylinder" size="1 1"/></body>
      <body pos="{position}"><freejoint/><geom name="box" type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    config=mink.Configuration(model)
    limit=RestoringCollisionAvoidanceLimit(model,geom_pairs=[(['box'],['cylinder'])],
        minimum_distance_from_collisions=.001,collision_detection_distance=5.)
    a,b=model.geom('box').id,model.geom('cylinder').id
    pair=np.empty(6);actual=limit.geom_distance(config.data,a,b,5.,pair)
    np.testing.assert_allclose(actual,expected,atol=5e-15,rtol=0.)
    np.testing.assert_allclose(np.linalg.norm(pair[:3]-pair[3:]),actual,atol=5e-15,rtol=0.)
    reverse=np.empty(6);assert limit.geom_distance(config.data,b,a,5.,reverse)==actual
    np.testing.assert_allclose(reverse,np.r_[pair[3:],pair[:3]],atol=5e-15)


def test_signed_penetration_remains_signed():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom name="cylinder" type="cylinder" size="1 1"/></body>
      <body pos="1 0 0"><freejoint/><geom name="box" type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    config=mink.Configuration(model)
    limit=RestoringCollisionAvoidanceLimit(model,geom_pairs=[(['box'],['cylinder'])],
        minimum_distance_from_collisions=.001,collision_detection_distance=5.)
    assert limit.geom_distance(config.data,model.geom('box').id,model.geom('cylinder').id,5.)<0.


def test_curved_contact_derivative_stays_consistent_at_small_steps():
    root=Path(__file__).resolve().parents[1]
    fixture=json.loads((root/'tests/fixtures/mesh_cylinder_precision_pose.json').read_text())
    model=RobotRetarget._load_model_with_ground(None,str(root/'asset/robot/booster_k1/K1_22dof.xml'))
    config=mink.Configuration(model);q=np.asarray(fixture['qpos']);config.update(q)
    a,b=fixture['geom_a'],fixture['geom_b']
    limit=RestoringCollisionAvoidanceLimit(model,geom_pairs=[([a],[b])],
        minimum_distance_from_collisions=.001,collision_detection_distance=.1)
    from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
    points=np.empty(6);distance=limit.geom_distance(config.data,a,b,.1,points)
    jac=compute_contact_normal_jacobian(model,config.data,a,b,points,np.zeros(3),np.zeros((3,model.nv)),np.zeros((3,model.nv)))
    rng=np.random.default_rng(31);directions=rng.normal(size=(32,model.nv));directions/=np.linalg.norm(directions,axis=1)[:,None]
    for direction in directions:
        trial=q.copy();mujoco.mj_integratePos(model,trial,direction,1e-7);config.update(trial)
        derivative=(limit.geom_distance(config.data,a,b,.1)-distance)/1e-7
        assert abs(derivative-jac@direction)<2e-7
