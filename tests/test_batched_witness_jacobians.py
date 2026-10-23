"""Batched rigid-body witness derivatives agree with MuJoCo and finite differences."""
import mujoco
import mink
import numpy as np
from scripts.solver import RestoringCollisionAvoidanceLimit


def test_batched_material_witnesses_under_rotating_free_bodies():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="a"><freejoint/><geom name="a" type="box" size=".1 .2 .3"/></body>
      <body name="b" pos=".7 0 .3"><freejoint/><geom name="b" type="box" size=".1 .2 .3"/></body>
      </worldbody></mujoco>''')
    config=mink.Configuration(model);limit=RestoringCollisionAvoidanceLimit(model,
        geom_pairs=[(['a'],['b'])],minimum_distance_from_collisions=.001,
        collision_detection_distance=.03)
    rng=np.random.default_rng(20261023);local=rng.normal(size=(2,41,3))*.2
    normals=rng.normal(size=(41,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
    for _ in range(8):
        config.integrate_inplace(rng.normal(size=model.nv),.1)
        data=config.data
        points=[local[k]@data.xmat[k+1].reshape(3,3).T+data.xpos[k+1] for k in range(2)]
        batched=limit.witness_jacobians(data,1,2,*points,normals)
        direct=[]
        for a,b,n in zip(*points,normals):
            ja,jb=np.zeros((3,model.nv)),np.zeros((3,model.nv))
            mujoco.mj_jac(model,data,ja,None,a,1);mujoco.mj_jac(model,data,jb,None,b,2)
            direct.append(n@(jb-ja))
        np.testing.assert_allclose(batched,direct,atol=2e-15,rtol=2e-15)
        q=config.q.copy();direction=rng.normal(size=model.nv);eps=1e-7
        positions=[]
        for sign in [-1,1]:
            config.update(q);config.integrate_inplace(direction,sign*eps)
            positions.append([local[k]@data.xmat[k+1].reshape(3,3).T+data.xpos[k+1] for k in range(2)])
        finite=np.sum(normals*((positions[1][1]-positions[1][0])-(positions[0][1]-positions[0][0])),axis=1)/(2*eps)
        np.testing.assert_allclose(batched@direction,finite,atol=5e-9,rtol=1e-8)
        config.update(q)
