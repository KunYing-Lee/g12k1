"""R9 float64 constrained G1-to-K1 solver; no output post-processing."""
import mink

import qpsolvers

import mujoco

import numpy as np

import pickle

import time

import csv

import copy

import os

from tqdm import tqdm

from scipy.spatial.transform import Rotation as Rot

from scipy.spatial import ConvexHull

from scipy.linalg import null_space

from scipy.optimize import linprog


def project_root_trajectory_axes(qpos, source_root_position, free_qposadr, axes):
    """Project a floating base onto exact source world-trajectory coordinates."""
    axis_indices = {"x": 0, "y": 1, "z": 2}
    projected = np.asarray(qpos, dtype=np.float64).copy()
    source = np.asarray(source_root_position, dtype=np.float64)
    if source.shape != (3,):
        raise ValueError(f"source_root_position must have shape (3,), got {source.shape}")
    if not axes:
        raise ValueError("root trajectory constraint requires at least one axis")
    if len(set(axes)) != len(axes):
        raise ValueError(f"root trajectory constraint axes must be unique, got {axes}")
    for axis in axes:
        if axis not in axis_indices:
            raise ValueError(f"root trajectory constraint axis must be x/y/z, got {axis!r}")
        index = axis_indices[axis]
        projected[int(free_qposadr) + index] = source[index]
    return projected

def solve_equality_reduced_qp(problem, solver, **settings):
    """Solve in the equality null space and reconstruct original KKT duals.

    x = particular + N z satisfies A x = b before the active-set solve.
    This is an algebraic reformulation of every dense IK QP, not a retry,
    constraint relaxation, or change to its objective.
    """
    P, g = np.asarray(problem.P), np.asarray(problem.q)
    A, b = problem.A, problem.b
    n = len(g)
    if A is None or len(A) == 0:
        N, particular = np.eye(n), np.zeros(n)
    else:
        A, b = np.asarray(A), np.asarray(b)
        particular = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(abs(A @ particular-b), initial=0.) > settings.get('primal_tol', 1e-12):
            return qpsolvers.Solution(problem, found=False)
        N = null_space(A)
    G, h = problem.G, problem.h
    finite = None if h is None else np.isfinite(h)
    rows = None if G is None else np.asarray(G)[finite]
    bounds = None if h is None else np.asarray(h)[finite] - rows @ particular
    reduced = qpsolvers.Problem(P=N.T @ P @ N, q=N.T @ (g+P @ particular),
        G=None if rows is None else rows @ N, h=bounds)
    if N.shape[1]:
        answer = qpsolvers.solve_problem(reduced, solver=solver, **settings)
        if not answer.found:
            return qpsolvers.Solution(problem, found=False, extras=answer.extras)
        x = particular + N @ answer.x
        z = None if h is None else np.zeros(len(h))
        if z is not None: z[finite] = answer.z
    else:
        if bounds is not None and np.min(bounds, initial=0.) < -settings.get('primal_tol', 1e-12):
            return qpsolvers.Solution(problem, found=False)
        x = particular
        z = None if h is None else np.zeros(len(h))
    residual = P @ x + g
    if z is not None: residual += np.asarray(G).T @ z
    y = None if A is None else np.linalg.lstsq(A.T, -residual, rcond=None)[0]
    return qpsolvers.Solution(problem, found=True, x=x, y=y, z=z,
        obj=float(.5*x @ P @ x + g @ x))

def feasibility_normal_step(problem, radius, soft_rows=None):
    """Trust-region L1 restoration subproblem with exact linear equalities.

    Elastic variables describe linearized infeasibility only during restoration;
    they are never carried into the nominal problem or final KKT certificate.
    The LP minimizes sum(max(G d-h, 0)) inside the trust box.
    """
    finite = np.isfinite(problem.h)
    G, h = np.asarray(problem.G)[finite], np.asarray(problem.h)[finite]
    n, m = G.shape[1], len(h)
    soft = np.ones(m, dtype=bool) if soft_rows is None else np.asarray(soft_rows)[finite]
    elastic = np.eye(m)[:, soft]
    slack_count = int(soft.sum())
    A = None if problem.A is None else np.c_[problem.A, np.zeros((len(problem.b), slack_count))]
    result = linprog(np.r_[np.zeros(n), np.ones(slack_count)],
        A_ub=np.c_[G, -elastic], b_ub=h, A_eq=A, b_eq=problem.b,
        bounds=[(-radius, radius)]*n + [(0., None)]*slack_count, method='highs',
        options={'primal_feasibility_tolerance': 1e-10,
                 'dual_feasibility_tolerance': 1e-10})
    if not result.success:
        raise RuntimeError('Feasibility normal subproblem failed: '+result.message)
    delta = result.x[:n]
    predicted = float(np.maximum(-h, 0.).sum()-np.maximum(G@delta-h, 0.).sum())
    return delta, predicted

class CartesianPoseTask(mink.FrameTask):
    """Independent world-position and chordal SO(3) residuals.

    SE(3).log translation is coupled to orientation even with zero orientation
    cost. Cartesian observations instead require Euclidean point residuals.
    The chordal rotation residual is smooth at pi and agrees to first order
    with rotation-vector distance; it introduces no temporal smoothing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost = np.r_[self.cost[:3], np.repeat(self.cost[3:], 3)]

    def _world_frame(self, configuration):
        # The objective is expressed in world coordinates. Reading MuJoCo's
        # existing transforms avoids matrix->quaternion->matrix round trips.
        kinds = {'body': (mujoco.mjtObj.mjOBJ_BODY, 'xpos', 'xmat', mujoco.mj_jacBody),
                 'geom': (mujoco.mjtObj.mjOBJ_GEOM, 'geom_xpos', 'geom_xmat', mujoco.mj_jacGeom),
                 'site': (mujoco.mjtObj.mjOBJ_SITE, 'site_xpos', 'site_xmat', mujoco.mj_jacSite)}
        kind, pos, mat, jac = kinds[self.frame_type]
        if getattr(self, '_model_identity', None) != id(configuration.model):
            self._resolved_frame = (int(self.frame_name) if isinstance(self.frame_name, (int, np.integer))
                else mujoco.mj_name2id(configuration.model, kind, self.frame_name))
            if self._resolved_frame < 0:
                raise ValueError('Unknown frame: '+str(self.frame_name))
            self._model_identity = id(configuration.model)
        fid = self._resolved_frame
        return (getattr(configuration.data, pos)[fid],
                getattr(configuration.data, mat)[fid].reshape(3,3), jac, fid)

    def compute_error(self, configuration):
        position, rotation, _, _ = self._world_frame(configuration)
        target = self.transform_target_to_world
        return np.r_[target.translation() - position,
            ((target.rotation().as_matrix() - rotation).T / np.sqrt(2)).ravel()]

    def compute_jacobian(self, configuration):
        _, rotation, jacobian, fid = self._world_frame(configuration)
        jac = np.empty((6, configuration.model.nv))
        jacobian(configuration.model, configuration.data, jac[:3], jac[3:], fid)
        derivative = np.cross(jac[3:].T[:, None, :], rotation.T[None, :, :])
        return np.vstack((-jac[:3], -derivative.reshape(configuration.model.nv, 9).T / np.sqrt(2)))

class WholeBodyGroundLimit(mink.Limit):
    """Exact support points against a horizontal half-space, in delta-q units."""

    def __init__(self, model, height=0.0):
        self.model, self.height = model, height
        self.ids = [g for g in range(model.ngeom)
            if (model.geom_contype[g] or model.geom_conaffinity[g])
            and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
        self.vertices = {}
        signs = np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for g in self.ids:
            typ = model.geom_type[g]
            if typ == mujoco.mjtGeom.mjGEOM_BOX:
                self.vertices[g] = signs * model.geom_size[g]
            elif typ == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]; start = model.mesh_vertadr[mid]
                vertices = model.mesh_vert[start:start+model.mesh_vertnum[mid]].astype(float)
                self.vertices[g] = vertices[ConvexHull(vertices).vertices]

    def points(self, data, g):
        m = self.model; rotation = data.geom_xmat[g].reshape(3,3)
        if g in self.vertices:
            return self.vertices[g] @ rotation.T + data.geom_xpos[g]
        typ, size, down = m.geom_type[g], m.geom_size[g], -rotation[2]
        if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            local = [size[0] * down]
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            local = [size[0]*down + np.array([0,0,z*size[1]]) for z in (-1,1)]
        elif typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            local = [size**2 * down / np.linalg.norm(size*down)]
        elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            norm = np.linalg.norm(down[:2])
            radial = [size[0]*down[:2]/norm] if norm > 1e-12 else [np.array([x,y])*size[0] for x,y in [(1,0),(-1,0),(0,1),(0,-1)]]
            local = [np.r_[xy,z*size[1]] for xy in radial for z in (-1,1)]
        else:
            raise ValueError(f'Unsupported ground collision geometry: {typ}')
        return np.asarray(local) @ rotation.T + data.geom_xpos[g]

    def gaps(self, configuration):
        return np.array([self.points(configuration.data,g)[:,2].min()-self.height for g in self.ids])

    def compute_qp_inequalities(self, configuration, dt):
        rows, bounds = [], []
        jac = np.zeros((3,self.model.nv))
        for g in self.ids:
            points = self.points(configuration.data,g)
            for point in points[points[:,2] < self.height + 0.03]:
                mujoco.mj_jac(self.model,configuration.data,jac,None,point,self.model.geom_bodyid[g])
                rows.append(-jac[2].copy()); bounds.append(point[2]-self.height)
        return mink.Constraint(G=np.asarray(rows).reshape(-1,self.model.nv),h=np.asarray(bounds))

class RootTranslationEquality(mink.Task):
    """Exact world-translation constraint in the differential IK problem."""

    def __init__(self, model, joint_id, axes):
        super().__init__(cost=np.ones(len(axes)))
        axis_ids = np.array([{"x": 0, "y": 1, "z": 2}[a] for a in axes])
        self.qpos_ids = model.jnt_qposadr[joint_id] + axis_ids
        self.jacobian = np.eye(model.nv)[model.jnt_dofadr[joint_id] + axis_ids]
        self.target = np.zeros(len(axes))

    def compute_error(self, configuration):
        return configuration.data.qpos[self.qpos_ids] - self.target

    def compute_jacobian(self, configuration):
        return self.jacobian

class TangentTrustRegionLimit(mink.Limit):
    """Per-iteration box trust region in MuJoCo tangent coordinates."""

    def __init__(
        self,
        model,
        max_free_translation_step,
        max_free_rotation_step,
        max_joint_step,
    ):
        step = np.full(model.nv, float(max_joint_step), dtype=np.float64)
        for joint_id in range(model.njnt):
            joint_type = int(model.jnt_type[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                step[dof_address : dof_address + 3] = float(
                    max_free_translation_step
                )
                step[dof_address + 3 : dof_address + 6] = float(
                    max_free_rotation_step
                )
            elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                step[dof_address : dof_address + 3] = float(max_joint_step)
        if not np.isfinite(step).all() or np.any(step <= 0.0):
            raise ValueError("trust-region steps must be finite and positive")
        self.step = step

    def compute_qp_inequalities(self, configuration, dt):
        del configuration, dt
        identity = np.eye(self.step.size, dtype=np.float64)
        return mink.Constraint(
            G=np.vstack((identity, -identity)),
            h=np.concatenate((self.step, self.step)),
        )

class FrameTrustRegionLimit(mink.Limit):
    """Bound the accumulated tangent displacement within one motion frame."""

    def __init__(
        self,
        model,
        max_free_translation_step,
        max_free_rotation_step,
        max_joint_step,
    ):
        self.model = model
        self.step = TangentTrustRegionLimit(
            model,
            max_free_translation_step=max_free_translation_step,
            max_free_rotation_step=max_free_rotation_step,
            max_joint_step=max_joint_step,
        ).step
        self.reference_qpos = model.qpos0.copy()
        self._displacement = np.empty(model.nv, dtype=np.float64)

    def set_reference(self, qpos):
        reference = np.asarray(qpos, dtype=np.float64)
        if reference.shape != (self.model.nq,):
            raise ValueError(
                f"frame trust-region reference must have shape ({self.model.nq},)"
            )
        self.reference_qpos = reference.copy()

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        mujoco.mj_differentiatePos(
            self.model,
            self._displacement,
            1.0,
            self.reference_qpos,
            configuration.data.qpos,
        )
        identity = np.eye(self.model.nv, dtype=np.float64)
        return mink.Constraint(
            G=np.vstack((identity, -identity)),
            h=np.concatenate(
                (
                    self.step - self._displacement,
                    self.step + self._displacement,
                )
            ),
        )

class RestoringCollisionAvoidanceLimit(mink.CollisionAvoidanceLimit):
    """Collision limit that actively restores the configured clearance."""

    def __init__(self, *args, recovery_gain=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_gain = float(recovery_gain)
        if not 0.0 < self.recovery_gain <= 1.0:
            raise ValueError("collision recovery_gain must be in (0, 1]")
        self._restoration_fromto = np.empty(6, dtype=np.float64)
        self._cap_pairs = {}
        self._polytope_pairs = {}
        self._polytopes = {}
        polytope_types = {int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX)}
        for row, pair in enumerate(self.geom_id_pairs):
            if all(int(self.model.geom_type[g]) in polytope_types for g in pair):
                self._polytope_pairs[row] = pair
            for polytope, cylinder in (pair, pair[::-1]):
                if (int(self.model.geom_type[polytope]) in polytope_types
                        and self.model.geom_type[cylinder] == mujoco.mjtGeom.mjGEOM_CYLINDER):
                    self._cap_pairs[row] = (polytope, cylinder)

    def geom_distance(self, data, a, b, cutoff, witness=None):
        """Signed distance with exact separated polytope/cylinder features.

        GJK screens separated pairs outside the requested range; EPA still
        supplies penetration depth. For a separated polytope/cylinder pair,
        enumerate the complete boundary feature set instead of differentiating
        a prematurely stagnated curved-support GJK simplex.
        """
        cached = getattr(self, '_geom_query_cache', None)
        if cached is None or not np.array_equal(cached[0], data.qpos):
            cached = (data.qpos.copy(), {})
            self._geom_query_cache = cached
        key = (a, b, cutoff)
        if key not in cached[1]:
            m = self.model
            points = np.zeros(6)
            distance = mujoco.mj_geomDistance(m, data, a, b, cutoff, points)
            polytypes = {int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX)}
            cylinder_type = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            polytope, cylinder = a, b
            flip = int(m.geom_type[a]) == cylinder_type and int(m.geom_type[b]) in polytypes
            if flip:
                polytope, cylinder = b, a
            if (0. < distance < cutoff and int(m.geom_type[polytope]) in polytypes
                    and int(m.geom_type[cylinder]) == cylinder_type):
                if __package__:
                    from .geometry import separated_polytope_cylinder_distance
                else:
                    from geometry import separated_polytope_cylinder_distance
                vertices, edges, faces, normals = self.polytope(polytope)
                rp = data.geom_xmat[polytope].reshape(3, 3)
                rc = data.geom_xmat[cylinder].reshape(3, 3)
                local = (vertices @ rp.T + (data.geom_xpos[polytope]-data.geom_xpos[cylinder])) @ rc
                local_normals = normals @ rp.T @ rc
                distance, pp, pc = separated_polytope_cylinder_distance(
                    local, edges, faces, local_normals, *m.geom_size[cylinder, :2])
                if distance <= 0.:
                    raise RuntimeError('Separated cylinder query has an intersecting feature')
                pair = np.r_[rc @ pp+data.geom_xpos[cylinder], rc @ pc+data.geom_xpos[cylinder]]
                points = np.r_[pair[3:], pair[:3]] if flip else pair
            cached[1][key] = (float(distance), points)
        distance, points = cached[1][key]
        if witness is not None:
            witness[:] = points
        return distance

    def contact_queries(self, configuration):
        data = configuration.data
        cached = getattr(self, '_contact_cache', None)
        if cached is not None and np.array_equal(cached[0], data.qpos):
            return cached[1:]
        distances = np.full(self.max_num_contacts, self.collision_detection_distance)
        witnesses = np.zeros((self.max_num_contacts, 6))
        indices = (self._broadphase_survivors(data)
                   if self.broadphase and self.max_num_contacts >= self.broadphase_min_pairs
                   else range(self.max_num_contacts))
        for row in indices:
            a, b = self.geom_id_pairs[row]
            distances[row] = self.geom_distance(data, a, b,
                self.collision_detection_distance, witnesses[row])
        self._contact_cache = (data.qpos.copy(), distances, witnesses)
        self._distance_cache = (data.qpos.copy(), distances)
        return distances, witnesses

    def witness_jacobians(self, data, body_a, body_b, points_a, points_b, normals):
        """Batch exact material-point derivatives from rigid-body twists.

        J(point) = J(origin) + omega cross (point-origin). All witnesses on
        a body share the same twist Jacobian; no geometry or row is removed.
        Cache only at exactly the same generalized position, in float64.
        """
        points_a, points_b, normals = map(lambda x: np.asarray(x, dtype=np.float64).reshape(-1, 3),
                                         (points_a, points_b, normals))
        if not len(normals):
            return np.empty((0, self.model.nv), dtype=np.float64)
        cache = getattr(self, '_body_twist_cache', None)
        if cache is None or not np.array_equal(cache[0], data.qpos):
            cache = (data.qpos.copy(), {})
            self._body_twist_cache = cache
        def projected(body, points):
            if body not in cache[1]:
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, data, jp, jr, body)
                cache[1][body] = jp, jr
            jp, jr = cache[1][body]
            return normals @ jp + np.cross(points-data.xpos[body], normals) @ jr
        return projected(body_b, points_b)-projected(body_a, points_a)

    def cap_feature_constraints(self, configuration, mesh, cylinder):
        """Linearize all mesh-edge witnesses on either flat cylinder cap.

        At parallel edge/face contact the distance has multiple minimizers;
        differentiating one arbitrary GJK witness misses other directional
        derivatives. Clip convex-hull edges against the cap disk and retain
        both interval endpoints. Each is a material point of the mesh, so
        its distance to the cylinder is a necessary (redundant) constraint,
        never an inflated or modified collision geometry.
        """
        m, data = self.model, configuration.data
        vertices, edges, _, _ = self.polytope(mesh)
        rm = data.geom_xmat[mesh].reshape(3, 3)
        rc = data.geom_xmat[cylinder].reshape(3, 3)
        world = vertices @ rm.T + data.geom_xpos[mesh]
        local = (world - data.geom_xpos[cylinder]) @ rc
        u, v = local[edges[:, 0]], local[edges[:, 1]]
        direction = v - u
        radius, height = m.geom_size[cylinder, :2]
        aa = np.sum(direction[:, :2]**2, axis=1)
        bb = 2 * np.sum(u[:, :2] * direction[:, :2], axis=1)
        cc = np.sum(u[:, :2]**2, axis=1) - radius**2
        discriminant = bb**2 - 4*aa*cc
        moving = aa > 0
        valid = (moving & (discriminant >= 0)) | (~moving & (cc <= 0))
        lo, hi = np.zeros(len(edges)), np.ones(len(edges))
        idx = np.flatnonzero(moving & valid)
        root = np.sqrt(discriminant[idx])
        lo[idx] = np.maximum(0., (-bb[idx]-root)/(2*aa[idx]))
        hi[idx] = np.minimum(1., (-bb[idx]+root)/(2*aa[idx]))
        valid &= lo <= hi
        points = np.vstack((u[valid] + lo[valid, None]*direction[valid],
                            u[valid] + hi[valid, None]*direction[valid]))
        points = np.unique(points, axis=0)
        distance = np.abs(points[:, 2]) - height
        closest = self.geom_distance(data, mesh, cylinder, self.collision_detection_distance)
        # The constraint is the minimum signed distance. Its generalized
        # Jacobian contains only minimizing witnesses, as for polytope faces.
        # Non-minimizing edge/disk intersections move with the configuration;
        # freezing them as material points creates additional restrictions
        # that are not derivatives of this distance constraint.
        keep = (distance >= 0) & (distance < self.collision_detection_distance) & (distance <= closest+1e-7)
        points, distance = points[keep], distance[keep]
        normals = np.sign(points[:, 2, None])*rc[:, 2]
        pmesh = points@rc.T+data.geom_xpos[cylinder]
        pcylinder = pmesh-distance[:, None]*normals
        rows = self.witness_jacobians(data, m.geom_bodyid[mesh], m.geom_bodyid[cylinder],
                                     pmesh, pcylinder, normals)
        gap = distance-self.minimum_distance_from_collisions
        return rows, np.where(gap >= 0., self.gain, self.recovery_gain)*gap

    def barrel_feature_constraints(self, configuration, polytope, cylinder):
        """Extremal witnesses of a cylinder generator against a flat face.

        The closest set on a cylinder barrel can be a line segment. Clip its
        full generator to the opposing convex faces and linearize both ends,
        which supplies every directional minimum of that closest segment.
        """
        m,data=self.model,configuration.data;rows,bounds=[],[];witness=np.zeros(6)
        dist=self.geom_distance(data,polytope,cylinder,self.collision_detection_distance,witness)
        if dist<=0 or dist>=self.collision_detection_distance:return rows,bounds
        normal=(witness[:3]-witness[3:])/np.linalg.norm(witness[:3]-witness[3:])
        rc=data.geom_xmat[cylinder].reshape(3,3);rp=data.geom_xmat[polytope].reshape(3,3)
        local_normal=rc.T@normal
        if abs(local_normal[2])>1e-6:return rows,bounds
        radial=local_normal[:2]/np.linalg.norm(local_normal[:2])*m.geom_size[cylinder,0]
        line=np.array([np.r_[radial,z*m.geom_size[cylinder,1]] for z in (-1,1)])@rc.T+data.geom_xpos[cylinder]
        local=(line-data.geom_xpos[polytope])@rp
        vertices,_,faces,normals=self.polytope(polytope)
        active=(normals@(rp.T@(-normal)))>=1.-1e-8
        jp,jc=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        for face,n in zip(faces[active],normals[active]):
            triangle=vertices[face];distances=(local-triangle[0])@n
            projected=local-distances[:,None]*n;u,v=projected;direction=v-u
            lo,hi=0.,1.
            for k in range(3):
                inward=np.cross(n,triangle[(k+1)%3]-triangle[k])
                value=float((u-triangle[k])@inward);slope=float(direction@inward)
                if abs(slope)<=1e-15:
                    if value < -1e-12:hi=-1.
                elif slope>0:lo=max(lo,-value/slope)
                else:hi=min(hi,-value/slope)
            if lo>hi:continue
            nw=rp@n
            for fraction in (lo,hi):
                pc=line[0]+fraction*(line[1]-line[0]);pp=rp@(u+fraction*direction)+data.geom_xpos[polytope]
                distance=float((pc-pp)@nw)
                if distance<0 or distance>dist+1e-7:continue
                mujoco.mj_jac(m,data,jp,None,pp,m.geom_bodyid[polytope]);mujoco.mj_jac(m,data,jc,None,pc,m.geom_bodyid[cylinder])
                rows.append(nw@(jp-jc));gap=distance-self.minimum_distance_from_collisions
                bounds.append((self.gain if gap>=0 else self.recovery_gain)*gap)
        return rows,bounds

    def polytope(self, geom):
        if geom not in self._polytopes:
            m = self.model
            if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                vertices = np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)]) * m.geom_size[geom]
            else:
                mid = m.geom_dataid[geom]; start=m.mesh_vertadr[mid]
                vertices=m.mesh_vert[start:start+m.mesh_vertnum[mid]].astype(float)
            hull=ConvexHull(vertices);faces=hull.simplices.copy();normals=hull.equations[:,:3]
            for i,face in enumerate(faces):
                if np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]])@normals[i] < 0:
                    faces[i,[1,2]]=faces[i,[2,1]]
            edges=np.unique(np.sort(np.vstack((faces[:,:2],faces[:,1:],faces[:,[0,2]])),axis=1),axis=0)
            self._polytopes[geom]=(vertices,edges,faces,normals)
        return self._polytopes[geom]

    def polytope_feature_constraints(self, configuration, mesh, box):
        """Enumerate both edge/face orders of the convex-polytope contact.

        A closest pair can be a mesh edge over a box face OR a box edge
        over a mesh face. Clipping edges to triangular face prisms retains
        all extremal material-point derivatives of the closest-point set.
        Coplanar triangulation only adds redundant necessary constraints.
        """
        m,data=self.model,configuration.data;rows,bounds=[],[]
        ja,jb=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        witness=np.zeros(6)
        closest_distance=self.geom_distance(data,mesh,box,self.collision_detection_distance,witness)
        if closest_distance <= 0 or closest_distance >= self.collision_detection_distance:
            return rows,bounds
        separation=(witness[3:]-witness[:3])/np.linalg.norm(witness[3:]-witness[:3])
        for a,b in [(mesh,box),(box,mesh)]:
            va,edges,_,_=self.polytope(a);vb,_,faces,normals=self.polytope(b)
            ra=data.geom_xmat[a].reshape(3,3);rb=data.geom_xmat[b].reshape(3,3)
            local=(va@ra.T+data.geom_xpos[a]-data.geom_xpos[b])@rb
            u=local[edges[:,0]];direction=local[edges[:,1]]-u
            # A positive-distance closest face must have its outward normal
            # aligned with the separating direction. Other faces cannot
            # belong to this minimum-distance witness set.
            separating_local=rb.T @ (-separation if a==mesh else separation)
            active=(normals @ separating_local) >= 1.-1e-8
            faces,normals=faces[active],normals[active]
            if not len(faces): continue
            triangles=vb[faces]
            if __package__:
                from .geometry import polytope_face_witnesses
            else:
                from geometry import polytope_face_witnesses
            witnesses = polytope_face_witnesses(u, direction, triangles, normals, closest_distance)
            points, normals, distances = witnesses[:, :3], witnesses[:, 3:6], witnesses[:, 6]
            world_normals = normals@rb.T
            pa = points@rb.T+data.geom_xpos[b]
            pb = pa-distances[:, None]*world_normals
            rows.extend(self.witness_jacobians(data, m.geom_bodyid[a], m.geom_bodyid[b],
                                               pa, pb, world_normals))
            gap = distances-self.minimum_distance_from_collisions
            bounds.extend(np.where(gap >= 0., self.gain, self.recovery_gain)*gap)
        return rows,bounds

    def compute_qp_inequalities(self, configuration, dt):
        # Mink builds a QP in delta-q, not velocity. Its collision limit's
        # safe-side bound divides by dt; evaluate at unit time so both safe
        # and restoring bounds use the same tangent-displacement units.
        # One distance query supplies both the separating Jacobian and the
        # restoring bound. Preserve Mink's conservative broadphase exactly.
        from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
        model, data = self.model, configuration.data
        distances, witnesses = self.contact_queries(configuration)
        upper_bound = np.full(self.max_num_contacts, np.inf)
        matrix = np.zeros((self.max_num_contacts, model.nv))
        for row in np.flatnonzero(distances < self.collision_detection_distance):
            a, b = self.geom_id_pairs[row]
            dist = distances[row]
            self._fromto[:] = witnesses[row]
            if abs(dist-self.collision_detection_distance) < 1e-12:
                continue
            jac = compute_contact_normal_jacobian(model, data, a, b,
                self._fromto, self._normal, self._jac1, self._jac2)
            matrix[row] = (-1. if dist >= 0. else 1.)*jac
            gap = dist-self.minimum_distance_from_collisions
            upper_bound[row] = (self.gain*gap+self.bound_relaxation if gap > 0.
                else min(self.bound_relaxation, self.recovery_gain*gap))
        self._distance_cache = (data.qpos.copy(), distances)
        constraint = mink.Constraint(G=matrix, h=upper_bound)
        feature_rows, feature_bounds = [], []
        for row, (mesh, cylinder) in self._cap_pairs.items():
            if np.isfinite(constraint.h[row]):
                rows, bounds = self.cap_feature_constraints(configuration, mesh, cylinder)
                feature_rows.extend(rows)
                feature_bounds.extend(bounds)
                rows, bounds = self.barrel_feature_constraints(configuration, mesh, cylinder)
                feature_rows.extend(rows)
                feature_bounds.extend(bounds)
        for row, (mesh, box) in self._polytope_pairs.items():
            if np.isfinite(constraint.h[row]):
                rows, bounds = self.polytope_feature_constraints(configuration, mesh, box)
                feature_rows.extend(rows)
                feature_bounds.extend(bounds)
        if feature_rows:
            return mink.Constraint(G=np.vstack((constraint.G, feature_rows)),
                h=np.r_[upper_bound, feature_bounds])
        return mink.Constraint(G=constraint.G, h=upper_bound)

class RobotRetarget:
    solve_qp = staticmethod(solve_equality_reduced_qp)


    def __init__(
        self,
        model_path: str,
        keypoint_path: str,
        ik_match_table: dict,
        solver: str = "daqp",
        verbose: bool = False,
        damping: float = 1.0,
        joints_limit_offset_degrees: dict | None = None,
        contact_body_names: list | tuple | dict | None = None,
        contact_position_cost: float = 10.0,
        posture_regularization: dict | None = None,
        ground_nonpenetration: dict | None = None,
        contact_grounding: dict | None = None,
        root_trajectory_constraint: dict | None = None,
        self_collision_avoidance: dict | None = None,
        temporal_regularization: dict | None = None,
        trajectory_fitting: dict | None = None,
        max_ik_iterations: int = 50,
        ik_convergence_tolerance: float = 1e-3,
    ):
        self.xml_file = model_path
        self.keypoint_path = keypoint_path
        self.ik_match_table = ik_match_table
        self.model = self._load_model_with_ground(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.max_iter = int(max_ik_iterations)
        self.ik_convergence_tolerance = float(ik_convergence_tolerance)
        if self.max_iter < 0:
            raise ValueError("max_ik_iterations must be non-negative")
        if self.ik_convergence_tolerance < 0.0:
            raise ValueError("ik_convergence_tolerance must be non-negative")
        self.damping = damping
        self.joints_limit_offset_degrees = joints_limit_offset_degrees or {}
        self.contact_body_names = self._normalize_contact_body_names(contact_body_names)
        self.contact_position_cost = float(contact_position_cost)
        self.posture_regularization = posture_regularization or {}
        self.ground_nonpenetration = ground_nonpenetration or {}
        self.contact_grounding = contact_grounding or {}
        self.root_trajectory_constraint = root_trajectory_constraint or {}
        self.self_collision_avoidance = self_collision_avoidance or {}
        self.temporal_regularization = temporal_regularization or {}
        self.trajectory_fitting = trajectory_fitting or {}
        self.trajectory_diagnostics = []
        self.verbose = verbose
        self.solver = solver
        self.human_body_to_task = {}
        self.task_errors = {}
        self.result_pos = []
        self.contact_state_name_to_idx = {}
        self.keypoint_name_to_idx = {}
        self.body_name_to_source_keypoint = {}
        self.body_name_to_contact_task = {}
        self.contact_targets = []
        self.current_contact_points = []
        self.motion_tasks = []
        self.active_tasks = []
        self.regularization_tasks = []
        self.ground_contact_body_ids = []
        self.ground_contact_geom_ids = []
        self.ground_root_z_qposadr = None
        self.ground_height = 0.0
        self.ground_tolerance = 1e-9
        self.ground_projection_count = 0
        self.ground_max_projection = 0.0
        self.root_trajectory_qposadr = None
        self.root_trajectory_axes = ()
        self.source_root_positions = None
        self.self_collision_limit = None
        self.self_collision_minimum_distance = 0.0
        self.self_collision_tolerance = 1e-5
        self.self_collision_max_backtracks = 12
        self.collision_restoration_limits = []
        self.frame_trust_region_limit = None
        self.temporal_posture_task = None
        self.root_translation_equality = None
        self.solver_diagnostics = []

        self.robot_motor_names = {}

        self.setup_retarget_configuration()
        self.setup_self_collision_avoidance()
        self.setup_posture_regularization()
        self.setup_temporal_regularization()
        self.setup_ground_nonpenetration()
        self.whole_body_ground_limit = None
        if self.ground_nonpenetration.get("whole_body", False):
            self.whole_body_ground_limit = WholeBodyGroundLimit(self.model, self.ground_height)
            self.ik_limits.append(self.whole_body_ground_limit)
        self.load_keypoints()
        self.setup_root_trajectory_constraint()
        self.setup_contact_targets()

    def _normalize_contact_body_names(self, contact_body_names):
        if contact_body_names is None:
            return []
        if not isinstance(contact_body_names, (list, tuple)):
            raise TypeError("contact_body_names must be a list or tuple")
        return list(contact_body_names)

    def _load_model_with_ground(self, xml_file: str) -> mujoco.MjModel:
        """Load the robot XML and add a MuJoCo default-style ground (checker plane + skybox).

        The ground geom has collision disabled (contype/conaffinity=0) and serves only as a
        visual reference; it does not affect mink IK's pure kinematic solving.
        """
        spec = mujoco.MjSpec.from_file(xml_file)

        # Skip if ground/skybox assets already exist to avoid name conflicts
        existing_tex = {t.name for t in spec.textures}
        existing_mat = {m.name for m in spec.materials}

        if "skybox" not in existing_tex:
            spec.add_texture(
                name="skybox",
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[0.3, 0.5, 0.7],
                rgb2=[0.0, 0.0, 0.0],
                width=512,
                height=512,
            )
        if "groundplane" not in existing_tex:
            spec.add_texture(
                name="groundplane",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                rgb1=[0.2, 0.3, 0.4],
                rgb2=[0.1, 0.2, 0.3],
                width=300,
                height=300,
            )
        if "groundplane" not in existing_mat:
            spec.add_material(
                name="groundplane",
                textures=["", "groundplane"],
                texrepeat=[5, 5],
                texuniform=True,
                reflectance=0.2,
            )

        # Lighting: main directional light + angled fill light to avoid an overly dark scene
        spec.worldbody.add_light(
            pos=[0, 0, 20.0],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
        )
        spec.worldbody.add_light(
            pos=[4, 4, 6.0],
            dir=[-0.5, -0.5, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.1, 0.1, 0.1],
        )

        # Add the (infinite) ground plane, collision disabled, for visualization only
        ground = spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane",
            pos=[0, 0, 0],
        )
        ground.contype = 0
        ground.conaffinity = 0

        return spec.compile()

    def load_keypoints(self):
        with open(self.keypoint_path, "rb") as f:
                keypoints_data = pickle.load(f)
        self.keypoint_names = keypoints_data["keypoint_names"]
        self.keypoint_name_to_idx = {
            keypoint_name: idx for idx, keypoint_name in enumerate(self.keypoint_names)
        }
        self.keypoints_pos = keypoints_data["positions"] 
        self.keypoints_quat = keypoints_data["quaternions"]  
        self.contact_names = keypoints_data.get("contact_names", [])
        self.contact_state_name_to_idx = {
            contact_name: idx for idx, contact_name in enumerate(self.contact_names)
        }
        self.contact_seq = keypoints_data["contact_states"]
        self.source_joint_positions = keypoints_data.get("source_joint_positions")
        self.source_joint_names = keypoints_data.get("source_joint_names", [])
        if self.posture_regularization.get("reference") == "source" and self.source_joint_positions is None:
            raise ValueError("Source posture requires source_joint_positions in the replay payload")
        self.source_root_positions = keypoints_data.get("source_root_positions")
        self.fps = keypoints_data.get("fps", 30)
        self.time_step = 1.0 / self.fps
        self.num_frames = self.keypoints_pos.shape[0]
        self.num_keypoints = self.keypoints_pos.shape[1]

    def setup_root_trajectory_constraint(self):
        """Configure exact floating-base trajectory axes carried by robot replay."""
        config = self.root_trajectory_constraint
        if not config or not bool(config.get("enabled", False)):
            return
        axes = config.get("axes", [])
        if not isinstance(axes, list):
            raise TypeError("root_trajectory_constraint.axes must be a list")
        if "z" in axes and (
            self.ground_contact_body_ids or self.ground_contact_geom_ids
        ):
            raise ValueError(
                "root_trajectory_constraint on z conflicts with ground_nonpenetration"
            )
        if self.source_root_positions is None:
            raise ValueError(
                "root_trajectory_constraint requires source_root_positions in the keypoint payload"
            )
        source_root_positions = np.asarray(self.source_root_positions, dtype=np.float64)
        if source_root_positions.shape != (self.num_frames, 3):
            raise ValueError(
                "source_root_positions must match the keypoint frame count, got "
                f"{source_root_positions.shape} for {self.num_frames} frames"
            )
        if not np.isfinite(source_root_positions).all():
            raise ValueError("source_root_positions contains non-finite values")
        free_joint_ids = np.flatnonzero(
            self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        )
        if free_joint_ids.size != 1:
            raise ValueError(
                "root_trajectory_constraint requires exactly one floating-base free joint, "
                f"found {free_joint_ids.size}"
            )
        free_qposadr = int(self.model.jnt_qposadr[int(free_joint_ids[0])])
        project_root_trajectory_axes(
            self.configuration.data.qpos,
            source_root_positions[0],
            free_qposadr,
            axes,
        )
        self.root_trajectory_qposadr = free_qposadr
        self.root_trajectory_axes = tuple(axes)
        self.root_translation_equality = RootTranslationEquality(
            self.model, int(free_joint_ids[0]), axes
        )
        self.source_root_positions = source_root_positions

    def project_root_trajectory(self, frame_idx):
        """Apply the configured affine equality before evaluating the IK state."""
        if self.root_trajectory_qposadr is None:
            return
        qpos = project_root_trajectory_axes(
            self.configuration.data.qpos,
            self.source_root_positions[frame_idx],
            self.root_trajectory_qposadr,
            self.root_trajectory_axes,
        )
        self.configuration.update(q=qpos)
        if self.root_translation_equality is not None:
            self.root_translation_equality.target = qpos[self.root_translation_equality.qpos_ids].copy()

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks = []
        self.robot_frame_names = []  # Robot body name corresponding to each task

        # Apply offsets to the configured joint limits: raise the lower bound
        self._apply_joints_limit_offset()
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        # VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
        # self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        for keypoint_name, entry in self.ik_match_table.items():
            robot_frame, pos_weight, rot_weight = entry
            if pos_weight != 0 or rot_weight != 0:
                task = CartesianPoseTask(
                    frame_name=robot_frame,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                # Use the keypoint name as the key for indexing keypoint data in update_targets
                self.human_body_to_task[keypoint_name] = task
                self.body_name_to_source_keypoint[robot_frame] = keypoint_name

                self.tasks.append(task)
                self.robot_frame_names.append(robot_frame)
                self.task_errors[task] = []

        self.motion_tasks = list(self.tasks)
        self.active_tasks = list(self.motion_tasks)

    def setup_self_collision_avoidance(self):
        """Add a whole-robot non-penetration inequality to the IK problem.

        Collision geometry and MuJoCo's contype/conaffinity masks define the
        constraint surface. Mink excludes same-body, welded, and direct
        parent-child pairs, so this remains a model-level physical constraint
        rather than a motion- or frame-specific correction.
        """
        config = self.self_collision_avoidance
        if not config or not bool(config.get("enabled", False)):
            return

        # Clearance/Jacobian consistency requires the distance oracle to be
        # more accurate than the SQP feasibility and optimality tolerances.
        self.model.opt.ccd_tolerance = float(config.get("distance_tolerance", 1e-10))
        self.model.opt.ccd_iterations = int(config.get("distance_iterations", 100))
        gain = float(config.get("gain", 0.85))
        minimum_distance = float(config.get("minimum_distance", 0.0))
        detection_distance = float(config.get("detection_distance", 0.03))
        bound_relaxation = float(config.get("bound_relaxation", 0.0))
        broadphase = bool(config.get("broadphase", True))
        tolerance = float(config.get("tolerance", 1e-5))
        max_backtracks = int(config.get("max_backtracks", 12))
        recovery_gain = float(config.get("recovery_gain", 0.5))
        trust_region = config.get("trust_region", {})
        if not isinstance(trust_region, dict):
            raise TypeError("self_collision_avoidance.trust_region must be a mapping")
        if not 0.0 < gain <= 1.0:
            raise ValueError("self_collision_avoidance.gain must be in (0, 1]")
        if detection_distance < minimum_distance:
            raise ValueError(
                "self_collision_avoidance.detection_distance must be greater than "
                "or equal to minimum_distance"
            )
        if bound_relaxation < 0.0:
            raise ValueError(
                "self_collision_avoidance.bound_relaxation must be non-negative"
            )
        if tolerance < 0.0:
            raise ValueError("self_collision_avoidance.tolerance must be non-negative")
        if max_backtracks < 0:
            raise ValueError(
                "self_collision_avoidance.max_backtracks must be non-negative"
            )

        # Select every collision-enabled robot geom. The injected ground plane
        # has zero collision masks and is therefore deliberately absent.
        geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_contype[geom_id]) != 0
            or int(self.model.geom_conaffinity[geom_id]) != 0
        ]
        if len(geom_ids) < 2:
            raise ValueError(
                "self_collision_avoidance requires at least two collision-enabled geoms"
            )

        collision_limit = RestoringCollisionAvoidanceLimit(
            self.model,
            geom_pairs=[(geom_ids, geom_ids)],
            gain=gain,
            minimum_distance_from_collisions=minimum_distance,
            collision_detection_distance=detection_distance,
            bound_relaxation=bound_relaxation,
            broadphase=broadphase,
            recovery_gain=recovery_gain,
        )
        if not collision_limit.geom_id_pairs:
            raise ValueError(
                "self_collision_avoidance produced no eligible geom pairs; check the "
                "model collision masks and kinematic tree"
            )
        self.self_collision_limit = collision_limit
        self.self_collision_minimum_distance = minimum_distance
        self.self_collision_tolerance = tolerance
        self.self_collision_max_backtracks = max_backtracks
        trust_region_limit = TangentTrustRegionLimit(
            self.model,
            max_free_translation_step=float(
                trust_region.get("max_free_translation_step", 0.02)
            ),
            max_free_rotation_step=float(
                trust_region.get("max_free_rotation_step", 0.05)
            ),
            max_joint_step=float(trust_region.get("max_joint_step", 0.05)),
        )
        frame_trust_region = config.get("frame_trust_region", {})
        if not isinstance(frame_trust_region, dict):
            raise TypeError(
                "self_collision_avoidance.frame_trust_region must be a mapping"
            )
        self.frame_trust_region_limit = FrameTrustRegionLimit(
            self.model,
            max_free_translation_step=float(
                frame_trust_region.get("max_free_translation_step", 0.20)
            ),
            max_free_rotation_step=float(
                frame_trust_region.get("max_free_rotation_step", 0.35)
            ),
            max_joint_step=float(
                frame_trust_region.get("max_joint_step", 0.35)
            ),
        )
        self.ik_limits.append(trust_region_limit)
        if not config.get("disable_frame_bound", False):
            self.ik_limits.append(self.frame_trust_region_limit)
        self.ik_limits.append(collision_limit)
        self.collision_restoration_limits = [
            self.ik_limits[0],
            trust_region_limit,
            self.frame_trust_region_limit,
            collision_limit,
        ]
        if self.verbose:
            print(
                "Self-collision avoidance: "
                f"{len(collision_limit.geom_id_pairs)} geom pairs, "
                f"minimum distance {minimum_distance:.4f} m, "
                f"detection distance {detection_distance:.4f} m"
            )

    def minimum_self_collision_distance(self):
        """Return the minimum signed distance over configured self-collision pairs."""
        if self.self_collision_limit is None:
            return np.inf
        distances, _ = self.self_collision_limit.contact_queries(self.configuration)
        return float(np.min(distances))



    def setup_posture_regularization(self):
        """Add an optional low-priority joint posture task.

        Position/orientation tasks determine the retargeted motion.  A posture
        task only resolves redundant or under-actuated joint solutions toward
        the target model's neutral qpos, which is especially important for
        robots with fewer arm DoFs than the source robot.
        """
        config = self.posture_regularization
        if not config or not bool(config.get("enabled", False)):
            return

        joint_costs = config.get("joint_costs", {})
        if not isinstance(joint_costs, dict) or not joint_costs:
            raise ValueError("posture_regularization.joint_costs must be a non-empty mapping")

        cost_vector = np.zeros(self.model.nv, dtype=np.float64)
        for joint_name, raw_cost in joint_costs.items():
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                str(joint_name),
            )
            if joint_id < 0:
                raise ValueError(f"Posture regularization joint not found: {joint_name}")
            cost = float(raw_cost)
            if cost < 0.0:
                raise ValueError(f"Posture regularization cost must be non-negative: {joint_name}")
            cost_vector[self.model.jnt_dofadr[joint_id]] = cost

        posture_task = mink.PostureTask(
            self.model,
            cost=cost_vector,
            gain=float(config.get("gain", 1.0)),
            lm_damping=float(config.get("lm_damping", 1.0)),
        )
        posture_task.set_target(self.model.qpos0)
        self.tasks.append(posture_task)
        self.motion_tasks.append(posture_task)
        self.active_tasks.append(posture_task)
        self.regularization_tasks.append(posture_task)
        self.posture_task = posture_task

    def setup_temporal_regularization(self):
        """Bias each frame toward the previous solved actuated posture."""
        config = self.temporal_regularization
        if not config or not bool(config.get("enabled", False)):
            return
        cost = float(config.get("cost", 1.0))
        if cost < 0.0:
            raise ValueError("temporal_regularization.cost must be non-negative")
        temporal_task = mink.PostureTask(
            self.model,
            cost=cost,
            gain=float(config.get("gain", 1.0)),
            lm_damping=float(config.get("lm_damping", 1.0)),
        )
        temporal_task.set_target(self.configuration.data.qpos)
        self.tasks.append(temporal_task)
        self.motion_tasks.append(temporal_task)
        self.active_tasks.append(temporal_task)
        self.regularization_tasks.append(temporal_task)
        self.task_errors[temporal_task] = []
        self.temporal_posture_task = temporal_task

    def setup_ground_nonpenetration(self):
        """Configure the exact root-Z projection for a horizontal ground plane."""
        config = self.ground_nonpenetration
        if not config or not bool(config.get("enabled", False)):
            return

        body_names = config.get("bodies", [])
        geom_names = config.get("geoms", [])
        if not isinstance(body_names, list) or not isinstance(geom_names, list):
            raise TypeError("ground_nonpenetration bodies/geoms must be lists")
        if not body_names and not geom_names:
            raise ValueError(
                "ground_nonpenetration requires a non-empty bodies or geoms list"
            )
        for body_name in body_names:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(body_name),
            )
            if body_id < 0:
                raise ValueError(
                    f"ground_nonpenetration body is absent from the target model: {body_name}"
                )
            self.ground_contact_body_ids.append(int(body_id))
        for geom_name in geom_names:
            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                str(geom_name),
            )
            if geom_id < 0:
                raise ValueError(
                    f"ground_nonpenetration geom is absent from the target model: {geom_name}"
                )
            if int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError(
                    "ground_nonpenetration currently requires exact box geoms, "
                    f"got {geom_name}"
                )
            self.ground_contact_geom_ids.append(int(geom_id))

        free_joint_ids = np.flatnonzero(
            self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        )
        if free_joint_ids.size != 1:
            raise ValueError(
                "ground_nonpenetration requires exactly one floating-base free joint, "
                f"found {free_joint_ids.size}"
            )
        free_qposadr = int(self.model.jnt_qposadr[int(free_joint_ids[0])])
        self.ground_root_z_qposadr = free_qposadr + 2
        self.ground_height = float(config.get("height", 0.0))
        self.ground_tolerance = float(config.get("tolerance", 1e-9))
        if self.ground_tolerance < 0.0:
            raise ValueError("ground_nonpenetration.tolerance must be non-negative")




    def setup_contact_targets(self):
        if not self.contact_body_names or not self.contact_state_name_to_idx:
            return

        if len(self.contact_body_names) != len(self.contact_names):
            if self.verbose:
                print(
                    "[contact target] skip: robot contact_links count "
                    f"{len(self.contact_body_names)} != keypoint contact_names count {len(self.contact_names)}"
                )
            return

        for body_name, contact_name in zip(self.contact_body_names, self.contact_names):
            contact_state_idx = self.contact_state_name_to_idx.get(contact_name)
            if contact_state_idx is None:
                continue

            source_keypoint_name = self._resolve_body_source_keypoint(body_name)
            if source_keypoint_name is None:
                if self.verbose:
                    print(f"[contact target] could not find a source keypoint for {body_name}")
                continue

            task = self._ensure_contact_task(body_name)
            self.contact_targets.append(
                {
                    "body_name": body_name,
                    "contact_name": contact_name,
                    "contact_state_idx": contact_state_idx,
                    "source_keypoint_name": source_keypoint_name,
                    "locked_position": None,
                    "task": task,
                }
            )

        if self.verbose and self.contact_targets:
            summary = [
                f"{item['contact_name']}->{item['body_name']} (src={item['source_keypoint_name']})"
                for item in self.contact_targets
            ]
            print(f"[contact target] enabled: {summary}")

    def _resolve_body_source_keypoint(self, body_name):
        if body_name in self.keypoint_name_to_idx:
            return body_name
        return self.body_name_to_source_keypoint.get(body_name)

    def _ensure_contact_task(self, body_name):
        existing_task = self.body_name_to_contact_task.get(body_name)
        if existing_task is not None:
            return existing_task

        task = CartesianPoseTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=self.contact_position_cost,
            orientation_cost=0.0,
            lm_damping=1,
        )
        self.body_name_to_contact_task[body_name] = task
        self.tasks.append(task)
        self.robot_frame_names.append(body_name)
        self.task_errors[task] = []
        return task

    def _get_target_pose(self, frame_idx, keypoint_name):
        keypoint_idx = self.keypoint_name_to_idx[keypoint_name]
        target_pos = self.keypoints_pos[frame_idx, keypoint_idx]
        target_quat = self.keypoints_quat[frame_idx, keypoint_idx]
        return target_pos, target_quat

    def _get_body_pose(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found: {body_name}")
        return self.configuration.data.xpos[body_id], self.configuration.data.xquat[body_id]

    def _get_contact_locked_position(self, frame_idx, contact_target):
        contact_state_idx = contact_target["contact_state_idx"]
        source_keypoint_name = contact_target["source_keypoint_name"]

        target_pos, _ = self._get_target_pose(frame_idx, source_keypoint_name)
        in_contact = bool(self.contact_seq[frame_idx, contact_state_idx])
        if not in_contact:
            contact_target["locked_position"] = None
            return target_pos, False

        if contact_target["locked_position"] is None:
            body_pos, _ = self._get_body_pose(contact_target["body_name"])
            contact_target["locked_position"] = np.asarray(
                body_pos, dtype=np.float64
            ).copy()
        return contact_target["locked_position"], True
    
    def _apply_joints_limit_offset(self):
        """Add offsets to the lower/upper limits of matching joints per joints_limit_offset_degrees.

        The key is a joint-name substring (e.g. knee_joint / elbow_joint) and matches all joints
        in the model whose name contains that substring (e.g. left_knee_joint, right_knee_joint).
        The value is [lower_offset_deg, upper_offset_deg]: the lower offset is added to the lower
        bound and the upper offset to the upper bound (positive raises, negative lowers). A single
        scalar value is also supported (applied to the lower bound only).
        """
        if not self.joints_limit_offset_degrees:
            return

        # Collect all joint names in the model
        all_joint_names = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None:
                all_joint_names.append((joint_id, name))

        for key, offset_deg in self.joints_limit_offset_degrees.items():
            # Support [lower_offset, upper_offset] or a single scalar (lower bound only)
            if isinstance(offset_deg, (list, tuple)):
                if len(offset_deg) != 2:
                    raise ValueError(
                        f"joints_limit_offset_degrees['{key}'] must be two values [lower, upper], "
                        f"got {offset_deg}"
                    )
                lower_offset_rad = np.radians(float(offset_deg[0]))
                upper_offset_rad = np.radians(float(offset_deg[1]))
            else:
                lower_offset_rad = np.radians(float(offset_deg))
                upper_offset_rad = 0.0

            if lower_offset_rad == 0.0 and upper_offset_rad == 0.0:
                continue
            matched = [(jid, name) for jid, name in all_joint_names if key in name]
            if not matched:
                if self.verbose:
                    print(f"[joint limit] no matching joint found: {key}")
                continue
            for joint_id, joint_name in matched:
                lower, upper = self.model.jnt_range[joint_id]
                new_lower = lower + lower_offset_rad
                new_upper = upper + upper_offset_rad
                # Ensure the lower bound does not exceed the upper bound
                if new_lower > new_upper:
                    new_lower, new_upper = new_upper, new_lower
                self.model.jnt_range[joint_id, 0] = new_lower
                self.model.jnt_range[joint_id, 1] = new_upper
                if self.verbose:
                    print(
                        f"[joint limit] {joint_name} range: "
                        f"[{lower:.4f}, {upper:.4f}] -> [{new_lower:.4f}, {new_upper:.4f}] rad"
                    )
 
    def update_targets(self, frame_idx):
        # Record the keypoint targets (pos, quat_wxyz) mapped to tasks for the current frame, for visualization
        self.current_targets = []
        self.current_contact_points = []
        self.active_tasks = list(self.motion_tasks)
        if self.posture_regularization.get("reference") == "source":
            target = self.model.qpos0.copy()
            for target_name, mapping in self.posture_regularization["joint_map"].items():
                source_name, scale, offset_degrees = mapping
                source_index = self.source_joint_names.index(source_name)
                address = self.model.joint(target_name).qposadr[0]
                target[address] = scale * self.source_joint_positions[frame_idx, source_index] + np.deg2rad(offset_degrees)
            self.posture_task.set_target(target)
        for keypoint_name, task in self.human_body_to_task.items():
            target_pos, target_quat = self._get_target_pose(frame_idx, keypoint_name)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos))
            robot_frame, pos_weight, rot_weight = self.ik_match_table[keypoint_name]
            self.current_targets.append(
                {
                    "pos": np.asarray(target_pos, dtype=np.float64),
                    "quat": np.asarray(target_quat, dtype=np.float64),
                    "robot_frame": robot_frame,
                    "pos_weight": float(pos_weight),
                    "rot_weight": float(rot_weight),
                }
            )

        for contact_target in self.contact_targets:
            source_keypoint_name = contact_target["source_keypoint_name"]
            target_pos, is_active = self._get_contact_locked_position(frame_idx, contact_target)
            if not is_active:
                continue

            _, target_quat = self._get_target_pose(frame_idx, source_keypoint_name)
            contact_target["task"].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos)
            )
            self.active_tasks.append(contact_target["task"])
            self.current_contact_points.append(np.asarray(target_pos, dtype=np.float64))
    
    def ik_equalities(self):
        constraint = getattr(self, "root_translation_equality", None)
        return [] if constraint is None else [constraint]

    def error(self):
        """Weighted residual norm of the actual objective, including posture.

        Zero-cost coordinates must not influence convergence. In particular,
        orientation residuals of position-only arm tasks are not objectives.
        """
        return float(np.sqrt(sum(
            np.sum(np.square(task.cost * task.compute_error(self.configuration)))
            for task in self.active_tasks
        )))

    def collision_violation(self):
        """L1 violation of the nominal nonlinear clearance constraints."""
        limit = self.self_collision_limit
        ground = getattr(self, "whole_body_ground_limit", None)
        ground_violation = 0.0 if ground is None else np.maximum(0., -ground.gaps(self.configuration)).sum()
        if limit is None:
            return float(ground_violation)
        distances, _ = limit.contact_queries(self.configuration)
        return float(ground_violation + np.maximum(0.,
            limit.minimum_distance_from_collisions-np.asarray(distances)).sum())


    def solve_frame(self, frame_idx, max_iterations=None):
        """Globalize SQP with an exact L1 constraint-penalty merit function.

        Clearance restoration may increase tracking error. The penalty is
        bounded below by the QP multipliers, so such a step is judged by its
        constrained optimization model rather than objective descent alone.
        """
        budget = self.max_iter if max_iterations is None else max_iterations
        dt = self.model.opt.timestep
        initial_error = error = self.error()
        reason, accepted, penalty = "iteration_budget", 0, 1.0
        radius = 0.05
        rejected = 0
        corrections = 0
        restorations = 0
        stationarity = complementarity = np.inf
        for iteration in range(budget):
            q0 = self.configuration.data.qpos.copy()
            problem = mink.build_ik(self.configuration, [], dt,
                0.0, limits=self.ik_limits, constraints=self.ik_equalities())
            for task in self.active_tasks:
                residual = task.cost * task.compute_error(self.configuration)
                jac = task.cost[:, None] * task.compute_jacobian(self.configuration)
                problem.P += jac.T @ jac
                problem.q += jac.T @ residual
            nominal_problem = qpsolvers.Problem(P=problem.P.copy(), q=problem.q.copy(), G=None if problem.G is None else problem.G.copy(), h=None if problem.h is None else problem.h.copy(), A=problem.A, b=problem.b)
            # A trust-region SQP re-solves the constrained direction when the
            # linear collision model is inaccurate. Scaling an old direction
            # also scales away the required normal restoration.
            original_rows = 0 if problem.G is None else len(problem.h)
            box = np.vstack((np.eye(self.model.nv), -np.eye(self.model.nv)))
            problem.G = box if problem.G is None else np.vstack((problem.G, box))
            bounds = np.full(2 * self.model.nv, radius)
            problem.h = bounds if problem.h is None else np.r_[problem.h, bounds]
            solution = self.solve_qp(problem, solver=self.solver, primal_tol=1e-12)
            if not solution.found:
                # A trust region can exclude every fully feasible linearized
                # step. Restore feasibility by reducing the L1 constraint
                # model; do not misreport this as an infeasible motion.
                violation = self.collision_violation()
                soft_rows = []
                for limit in self.ik_limits:
                    rows = limit.compute_qp_inequalities(self.configuration, dt)
                    count = 0 if rows.h is None else len(rows.h)
                    soft_rows.extend([isinstance(limit, (WholeBodyGroundLimit,
                        RestoringCollisionAvoidanceLimit))]*count)
                delta, predicted = feasibility_normal_step(nominal_problem, radius, soft_rows)
                if predicted <= 0. or violation <= 1e-12:
                    raise mink.NoSolutionFound(self.solver)
                self.configuration.integrate_inplace(delta, 1.)
                next_violation = self.collision_violation()
                ratio = (violation-next_violation)/predicted
                if ratio < .25:
                    radius *= .25
                elif ratio > .75 and np.max(abs(delta)) >= .9*radius:
                    radius *= 2.
                if ratio < 1e-4:
                    self.configuration.update(q=q0)
                    rejected += 1
                else:
                    restorations += 1
                    accepted += 1
                    error = self.error()
                continue
            delta = solution.x
            violation = self.collision_violation()
            stationarity_solution = solution
            if radius <= 10 * self.ik_convergence_tolerance:
                stationarity_solution = self.solve_qp(nominal_problem, solver=self.solver, primal_tol=1e-12)
            stationarity = np.inf
            complementarity = np.inf
            if stationarity_solution.found:
                dual_gradient = nominal_problem.q.copy()
                if stationarity_solution.z is not None and nominal_problem.G is not None:
                    dual_gradient += nominal_problem.G.T @ stationarity_solution.z[:original_rows]
                if stationarity_solution.y is not None and nominal_problem.A is not None:
                    dual_gradient += nominal_problem.A.T @ stationarity_solution.y
                stationarity = np.linalg.norm(dual_gradient, ord=np.inf) / max(1., np.linalg.norm(nominal_problem.q, ord=np.inf))
                complementarity = 0.0
                if nominal_problem.h is not None and stationarity_solution.z is not None:
                    finite = np.isfinite(nominal_problem.h)
                    complementarity = np.max(np.abs(stationarity_solution.z[:original_rows][finite] * nominal_problem.h[finite]), initial=0.) / max(1., .5*error**2)
            if (stationarity <= self.ik_convergence_tolerance
                    and complementarity <= self.ik_convergence_tolerance
                    and violation <= 1e-8):
                reason = "kkt_tolerance"
                break
            if solution.z is not None and original_rows:
                penalty = max(penalty, 2.0 * float(np.max(np.abs(solution.z[:original_rows]))))
            # The linearized collision constraints restore at least this
            # fraction of each violation. Other bounds are linear in delta-q.
            gain = self.self_collision_limit.recovery_gain if self.self_collision_limit else 1.0
            directional = float(problem.q @ delta) - penalty * gain * violation
            if violation > 0 and directional >= 0:
                penalty = max(penalty, 2.0 * max(0.0, float(problem.q @ delta)) / (gain * violation))
                directional = float(problem.q @ delta) - penalty * gain * violation
            merit = 0.5 * error**2 + penalty * violation
            self.configuration.integrate_inplace(delta, 1.0)
            next_error = self.error()
            next_violation = self.collision_violation()
            if (next_violation > violation and
                    .5*next_error**2 + penalty*next_violation > merit + 1e-4*directional):
                # SQP second-order correction (Maratos correction): restore
                # nonlinear clearance at the trial point while retaining the
                # tangential optimization step. It is part of this solve and
                # is accepted only by the same merit/trust-region test.
                correction_problem = mink.build_ik(self.configuration, [], dt,
                    1.0, limits=self.ik_limits, constraints=self.ik_equalities())
                if self.self_collision_limit is not None:
                    # Locate its rows without assuming a particular order
                    # of configuration, frame, and collision limits.
                    offset = 0
                    for limit in self.ik_limits:
                        rows = limit.compute_qp_inequalities(self.configuration, dt)
                        n = 0 if rows.h is None else len(rows.h)
                        if limit is self.self_collision_limit:
                            h = correction_problem.h[offset:offset+n]
                            h[h < 0] /= gain
                            break
                        offset += n
                correction = self.solve_qp(correction_problem,
                    solver=self.solver, primal_tol=1e-12)
                if correction.found and np.linalg.norm(correction.x, ord=np.inf) <= np.linalg.norm(delta, ord=np.inf):
                    corrections += 1
                    self.configuration.integrate_inplace(correction.x, 1.0)
                    next_error = self.error()
                    next_violation = self.collision_violation()
            next_merit = 0.5 * next_error**2 + penalty * next_violation
            predicted_reduction = -directional - 0.5 * float(delta @ problem.P @ delta)
            ratio = ((merit - next_merit) / predicted_reduction
                     if predicted_reduction > 0 else -np.inf)
            if ratio < 0.25:
                radius *= 0.25
            elif ratio > 0.75 and np.linalg.norm(delta, ord=np.inf) >= 0.9 * radius:
                radius *= 2.0
            if ratio < 1e-4:
                self.configuration.update(q=q0)
                rejected += 1
                continue
            accepted += 1
            error = next_error

        self.solver_diagnostics.append(dict(
            frame=int(frame_idx), iterations=accepted, reason=reason,
            initial_weighted_error=initial_error, final_weighted_error=error,
            final_clearance_violation=self.collision_violation(), merit_penalty=penalty,
            rejected_trials=rejected, qp_solves=iteration + 1, final_trust_radius=radius,
            feasibility_restorations=restorations, second_order_corrections=corrections, normalized_stationarity=float(stationarity), normalized_complementarity=float(complementarity),
        ))

    def initialize_first_pose(self):
        """Fit frame zero before imposing any previous-frame prior or bound."""
        # The floating base is observed in the source; model.qpos0 is not an
        # observation and must not anchor the start of the exported trajectory.
        free_ids = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free_ids.size == 1:
            jid = int(free_ids[0])
            root_body = int(self.model.jnt_bodyid[jid])
            for keypoint, (body, _, _) in self.ik_match_table.items():
                if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body) == root_body:
                    pos, quat = self._get_target_pose(0, keypoint)
                    q = self.configuration.data.qpos.copy()
                    adr = int(self.model.jnt_qposadr[jid])
                    q[adr:adr+3], q[adr+3:adr+7] = pos, quat
                    self.configuration.update(q=q)
                    break
        self.project_root_trajectory(0)
        limits, motion_tasks, contacts = self.ik_limits, self.motion_tasks, self.contact_targets
        restoration_limits = self.collision_restoration_limits
        try:
            self.ik_limits = [x for x in limits if x is not self.frame_trust_region_limit]
            self.collision_restoration_limits = [x for x in restoration_limits if x is not self.frame_trust_region_limit]
            self.motion_tasks = [x for x in motion_tasks if x is not self.temporal_posture_task]
            self.contact_targets = []
            self.update_targets(0)
            self.solve_frame(0, max_iterations=max(200, self.max_iter))
        finally:
            self.ik_limits, self.motion_tasks, self.contact_targets = limits, motion_tasks, contacts
            self.collision_restoration_limits = restoration_limits
        self.solver_diagnostics[-1]["phase"] = "initialization"



    
    def retarget(self):
        """Initialize every frame, then solve the complete constrained trajectory."""
        if self.whole_body_ground_limit is None or not self.trajectory_fitting.get("enabled"):
            raise ValueError("R9 requires whole-body ground constraints and trajectory fitting")
        initialization_started = time.perf_counter()
        self.stage_timings = {}
        self.initialize_first_pose()
        for frame_idx in tqdm(range(self.num_frames), desc="Retargeting", unit="frame"):
            # Root XY is an exact source-trajectory equality. Establish it
            # before anchoring the per-frame trust region; otherwise a
            # source clip starting away from the model origin makes the
            # second IK iteration infeasible.
            self.project_root_trajectory(frame_idx)
            if self.frame_trust_region_limit is not None:
                self.frame_trust_region_limit.set_reference(
                    self.configuration.data.qpos
                )
            if self.temporal_posture_task is not None:
                self.temporal_posture_task.set_target(
                    self.configuration.data.qpos
                )
            self.update_targets(frame_idx)

            self.solve_frame(frame_idx)
            curr_pos = self.configuration.data.qpos.copy()
            self.result_pos.append(curr_pos)

        self.stage_timings['frame_initialization_seconds'] = time.perf_counter()-initialization_started
        if self.trajectory_fitting.get("enabled", False):
            trajectory_started = time.perf_counter()
            if __package__:
                from .trajectory import TrajectoryRetarget
            else:
                from trajectory import TrajectoryRetarget
            trajectory = TrajectoryRetarget(self, self.result_pos,
                velocity_cost=float(self.trajectory_fitting["velocity_cost"]))
            try:
                result, converged = trajectory.solve(
                    max_iterations=int(self.trajectory_fitting.get("max_iterations", 200)),
                    tolerance=self.ik_convergence_tolerance)
            finally:
                self.trajectory_diagnostics = trajectory.history
                self.schwarz_diagnostics = trajectory.schwarz_diagnostics
                self.stage_timings['trajectory_seconds'] = time.perf_counter()-trajectory_started
            self.trajectory_candidate = result
            self.trajectory_converged = converged
            if not converged:
                raise RuntimeError("Joint trajectory optimization did not reach KKT tolerance")
            self.result_pos = list(result)
    
    def save_results_as_csv(self, output_path):
        if len(self.result_pos) == 0:
            raise ValueError("No retarget results to save. Run retarget() first.")

        # [num_frame, num_joint], the first 7 dims are posXYZ + quat(wxyz)
        result = np.asarray(self.result_pos, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] < 7:
            raise ValueError(
                f"result_pos shape must be [num_frame, num_joint>=7], got {result.shape}"
            )

        # Reorder the first-7-column quaternion from wxyz (cols 3..6) to xyzw
        result_xyzw = result.copy()
        result_xyzw[:, 3:7] = result[:, [4, 5, 6, 3]]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result_xyzw.tolist())

        if self.verbose:
            print(f"Saved retarget results to: {output_path} (shape={result_xyzw.shape})")
