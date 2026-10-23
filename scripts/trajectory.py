"""Sparse constrained trajectory fitting, with observed articulation velocities.

The temporal residual is D(q - source), not Dq: a feasible source trajectory
has zero velocity residual regardless of its frequency or amplitude. All poses
and collision constraints participate in the same nonlinear optimization.
"""
import copy
import time
import os
import sys
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.sparse as sp
import mink
import mujoco
import qpsolvers
if __package__:
    from .solver import TangentTrustRegionLimit
else:
    from solver import TangentTrustRegionLimit


def velocity_operator(count, nv, dofs, dt, cost):
    rows = np.arange((count - 1) * len(dofs))
    first = (np.repeat(np.arange(count - 1), len(dofs)) * nv
             + np.tile(dofs, count - 1))
    return sp.coo_matrix((np.r_[-np.full(len(rows), cost / dt),
                                np.full(len(rows), cost / dt)],
                         (np.r_[rows, rows], np.r_[first, first + nv])),
                        shape=(len(rows), count * nv)).tocsr()


def consolidate_coordinate_bounds(problem):
    """Exact intersection of scalar bounds, with an original-row dual map.

    Configuration bounds and both trust boxes often constrain the same
    coordinate three times. Their intersection is exactly the tightest bound.
    No geometric row, precision threshold, or feasible point is discarded.
    """
    G = problem.G.tocsr()
    scalar = np.flatnonzero(np.diff(G.indptr) == 1)
    general = np.flatnonzero(np.diff(G.indptr) != 1)
    coefficients = G.data[G.indptr[scalar]]
    columns = G.indices[G.indptr[scalar]]
    keys = 2*columns + (coefficients < 0)
    bounds = problem.h[scalar]/np.abs(coefficients)
    order = np.lexsort((scalar, bounds, keys))
    ordered_keys = keys[order]
    keep = order[np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]] if len(order) else order
    selected = np.r_[general, scalar[keep]]
    scales = np.r_[np.ones(len(general)), np.abs(coefficients[keep])]
    reduced = qpsolvers.Problem(P=problem.P, q=problem.q,
        G=(sp.diags(1./scales)@G[selected]).tocsc(), h=problem.h[selected]/scales,
        A=problem.A, b=problem.b)
    return reduced, selected, scales


class TrajectoryRetarget:
    def __init__(self, retargeter, poses, frames=None, velocity_cost=0.1):
        self.r = retargeter
        # This is a position-level SQP, not a differential-IK velocity damper.
        # Its linearized signed-distance constraints use the full distance.
        if self.r.self_collision_limit is not None:
            self.r.self_collision_limit.gain = 1.
            self.r.self_collision_limit.recovery_gain = 1.
        self.q = np.asarray(poses).copy()
        self.initialization_q = self.q.copy()
        self.frames = list(range(len(poses))) if frames is None else list(frames)
        self.tasks = []
        self.references = []
        mapping = self.r.posture_regularization['joint_map']
        self.qadr = np.array([self.r.model.joint(j).qposadr[0] for j in mapping])
        dofs = np.array([self.r.model.joint(j).dofadr[0] for j in mapping])
        self.dofs = dofs
        # Replay contact acquisition once. Task targets remain fixed throughout
        # the nonlinear solve, including all line searches and corrections.
        for contact in self.r.contact_targets:
            contact['locked_position'] = None
        for i, frame in enumerate(self.frames):
            self.r.configuration.update(q=self.q[max(0, i - 1)])
            self.r.project_root_trajectory(frame)
            self.r.update_targets(frame)
            self.tasks.append(copy.deepcopy(self.r.active_tasks))
            self.references.append([
                self.r.source_joint_positions[frame, self.r.source_joint_names.index(src)] * scale
                + np.deg2rad(offset) for src, scale, offset in mapping.values()])
        self.references = np.asarray(self.references)
        self.V = velocity_operator(len(self.q), self.r.model.nv, dofs,
                                   self.r.time_step, velocity_cost)
        self.velocity_scale = velocity_cost / self.r.time_step
        self.Hv = (self.V.T @ self.V).tocsc()
        self.history = []
        self._evaluation_cache = {}
        self._linearization_cache = {}
        self.frame_workers = int(os.environ.get('RETARGET_FRAME_WORKERS', '1'))
        if self.frame_workers < 1:
            raise ValueError('RETARGET_FRAME_WORKERS must be positive')
        if self.frame_workers > 1 and sys.platform != 'linux':
            raise ValueError('Parallel frame evaluation requires Linux fork workers')
        self._pool = None
        self.fixed_frames = set()
        self.is_subproblem = False
        self.schwarz_diagnostics = []

    def select(self, i, q):
        self.r.configuration.update(q=q)
        self.r.project_root_trajectory(self.frames[i])
        self.r.active_tasks = self.tasks[i]

    def velocity_residual(self, q):
        return (np.diff(q[:, self.qadr] - self.references, axis=0)
                * self.velocity_scale).ravel()

    def _frame_residual(self, i, pose):
        entries = self._evaluation_cache.setdefault(i, [])
        for cached in entries:
            if np.array_equal(cached[0], pose):
                return cached[1:]
        self.select(i, pose)
        residual = np.concatenate([task.cost*task.compute_error(self.r.configuration)
                                   for task in self.r.active_tasks])
        result = (pose.copy(), residual, self.r.collision_violation())
        entries.append(result)
        del entries[:-2]
        return result[1:]

    def _frame_evaluate(self, i, pose):
        if i in self.fixed_frames:
            return 0., 0.
        residual, violation = self._frame_residual(i, pose)
        return .5*float(residual@residual), violation

    def _frame_difference(self, i, pair):
        if i in self.fixed_frames:
            return 0.
        old, _ = self._frame_residual(i, pair[0])
        new, _ = self._frame_residual(i, pair[1])
        return .5*float((new-old)@(new+old))

    def objective_difference(self, old, new):
        # Algebraically identical to f(new)-f(old), without subtracting
        # whole-trajectory totals containing large constant residuals.
        from math import fsum
        change = fsum(self._frames('difference', np.stack((old, new), axis=1)))
        v0, v1 = self.velocity_residual(old), self.velocity_residual(new)
        return change + .5*float((v1-v0)@(v1+v0))

    def _frame_linearize(self, i, q):
        if i in self.fixed_frames:
            nv = self.r.model.nv
            return (sp.csc_matrix((nv,nv)), np.zeros(nv),
                sp.csc_matrix((0,nv)), np.zeros(0), sp.eye(nv,format='csc'), np.zeros(nv))
        cached = self._linearization_cache.get(i)
        if cached is None or not np.array_equal(cached[0], q):
            self.select(i, q)
            p = mink.build_ik(self.r.configuration, [], self.r.model.opt.timestep,
                             0., limits=[limit for limit in self.r.ik_limits
                                 if not isinstance(limit, TangentTrustRegionLimit)],
                             constraints=self.r.ik_equalities())
            for task in self.r.active_tasks:
                jac = task.cost[:, None] * task.compute_jacobian(self.r.configuration)
                residual = task.cost * task.compute_error(self.r.configuration)
                p.P += jac.T @ jac
                p.q += jac.T @ residual
            finite = np.isfinite(p.h)
            cached = (q.copy(), sp.csc_matrix(p.P), p.q,
                      sp.csc_matrix(p.G[finite]), p.h[finite],
                      sp.csc_matrix((0, self.r.model.nv)) if p.A is None else sp.csc_matrix(p.A),
                      np.zeros(0) if p.b is None else p.b)
            self._linearization_cache[i] = cached
        return cached[1:]

    def _frame_certificate(self, i, data):
        pose, gradient = data[:self.r.model.nq], data[self.r.model.nq:]
        _, _, G, h, A, _ = self._frame_linearize(i, pose)
        residual, dual = current_point_multipliers(gradient, G.toarray(), h, A.toarray())
        scale = max(1., float(np.max(abs(gradient))))
        return (float(np.max(abs(residual)))/scale,
                float(np.max(abs(dual*h), initial=0.)), float(np.max(abs(residual))))

    def _frames(self, kind, poses, max_norm=None):
        if self.frame_workers == 1:
            method = getattr(self, '_frame_'+kind)
            return [method(i, q, max_norm) if kind == 'correct' else method(i, q)
                    for i, q in enumerate(poses)]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.frame_workers,
                mp_context=multiprocessing.get_context('fork'),
                initializer=_initialize_frame_worker, initargs=(self,))
        chunks = np.array_split(np.arange(len(poses)), min(self.frame_workers, len(poses)))
        jobs = [(kind, int(ids[0]), poses[ids], max_norm) for ids in chunks]
        return [row for batch in self._pool.map(_frame_batch, jobs) for row in batch]

    def evaluate(self, q):
        cost, violation = 0., 0.
        for frame_cost, frame_violation in self._frames('evaluate', q):
            cost += frame_cost
            violation += frame_violation
        v = self.velocity_residual(q)
        return cost + .5 * v @ v, float(violation)

    def linearize(self):
        rows = self._frames('linearize', self.q)
        hessians, gradients, inequalities, bounds, equalities, rhs = zip(*rows)
        self.constraint_offsets = np.r_[0, np.cumsum([len(h) for h in bounds])]
        return qpsolvers.Problem(
            P=sp.block_diag(hessians, format='csc') + self.Hv,
            q=np.concatenate(gradients) + self.V.T @ self.velocity_residual(self.q),
            G=sp.block_diag(inequalities, format='csc'), h=np.concatenate(bounds),
            A=sp.block_diag(equalities, format='csc'), b=np.concatenate(rhs))

    @staticmethod
    def qp(problem):
        reduced, selected, scales = consolidate_coordinate_bounds(problem)
        solution = qpsolvers.solve_problem(reduced, solver='clarabel',
            tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-11, max_iter=200,
            max_threads=1)
        if not solution.found:
            raise RuntimeError('Trajectory QP did not converge')
        dual = np.zeros(len(problem.h))
        dual[selected] = solution.z/scales
        return qpsolvers.Solution(problem, found=True, x=solution.x,
            y=solution.y, z=dual, obj=solution.obj, extras=solution.extras)

    def integrate(self, delta):
        # Integration needs no frame transforms or Jacobians. Evaluate those
        # once at the trial pose, rather than three times per integration.
        result = self.q.copy()
        for i, (q, dq) in enumerate(zip(result, delta.reshape(len(result), -1))):
            if i in self.fixed_frames:
                continue
            mujoco.mj_integratePos(self.r.model, q, dq, 1.)
        return result

    def _frame_correct(self, i, q, max_norm):
        if i in self.fixed_frames:
            return q.copy()
        self.select(i, q)
        displacement = np.empty(self.r.model.nv)
        violation = self.r.collision_violation()
        # Distribute the unchanged aggregate feasibility tolerance across
        # frames; the sum of successful corrections meets the original gate.
        local_tolerance = 1e-8/len(self.frames)
        for _ in range(8):
            if violation <= local_tolerance:
                return self.r.configuration.q.copy()
            p = mink.build_ik(self.r.configuration, [], self.r.model.opt.timestep,
                             1., limits=self.r.ik_limits, constraints=self.r.ik_equalities())
            solution = self.r.solve_qp(p, solver=self.r.solver, primal_tol=1e-12)
            if not solution.found:
                return None
            self.r.configuration.integrate_inplace(solution.x, 1.)
            mujoco.mj_differentiatePos(self.r.model, displacement, 1.,
                                      q, self.r.configuration.q)
            next_violation = self.r.collision_violation()
            if np.max(abs(displacement)) > max_norm or next_violation >= violation:
                return None
            violation = next_violation
        return self.r.configuration.q.copy() if violation <= local_tolerance else None

    def correct(self, trial, max_norm):
        result = self._frames('correct', trial, max_norm)
        return trial if any(q is None for q in result) else np.asarray(result)

    def solve(self, max_iterations=200, tolerance=1e-6):
        try:
            if len(self.q) > 128:
                self.schwarz_sweep(128, max_iterations, tolerance)
            return self._solve(max_iterations, tolerance)
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None

    def schwarz_sweep(self, block_size, max_iterations, tolerance):
        """Multiplicative overlapping nonlinear Schwarz preconditioning.

        Each conditional problem retains the exact source-velocity terms to
        fixed neighboring frames. Three colors make simultaneously optimized
        blocks nonadjacent, so neither variables nor objective terms overlap.
        The complete trajectory is still certified by the global SQP below.
        """
        count = len(self.q)
        starts = list(range(0, count, block_size//2))
        if self.frame_workers > 1 and self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.frame_workers,
                mp_context=multiprocessing.get_context('fork'),
                initializer=_initialize_frame_worker, initargs=(self,))
        for color in range(3):
            jobs = []
            for index in range(color, len(starts), 3):
                start, end = starts[index], min(count, starts[index]+block_size)
                left, right = max(0,start-1), min(count,end+1)
                jobs.append((start,end,left,right,self.q[left:right].copy(),max_iterations,tolerance))
            results = ([_solve_block(self, job) for job in jobs] if self._pool is None
                       else list(self._pool.map(_block_worker, jobs)))
            for start, end, pose, report in results:
                if not report['converged']:
                    raise RuntimeError('Schwarz conditional problem did not converge: '+str(report))
                self.q[start:end] = pose
                self.schwarz_diagnostics.append(report)
            print({'schwarz_color':color,'blocks':len(jobs),'converged':True},flush=True)
            checkpoint = os.environ.get('RETARGET_TRAJECTORY_CHECKPOINT')
            if checkpoint:
                np.save(checkpoint+'.color-'+str(color)+'.npy', self.q)

    def _solve(self, max_iterations=200, tolerance=1e-6):
        """Exact L1 merit trust-region SQP with second-order corrections.

        Each frame's penalty exceeds the sum of its inequality multipliers.
        This bounds the multiplier of every grouped signed-distance defect,
        including contacts represented by several support-feature rows.
        The trust box bounds an optimizer update, never inter-frame motion.
        KKT certificates exclude the adaptive trust-box multipliers.
        """
        value, violation = self.evaluate(self.q)
        penalties = np.ones(len(self.q))
        violations = np.asarray([row[1] for row in self._frames("evaluate", self.q)])
        radius = .05
        static_step = np.full(self.r.model.nv, np.inf)
        for limit in self.r.ik_limits:
            if isinstance(limit, TangentTrustRegionLimit):
                static_step = np.minimum(static_step, limit.step)
        for iteration in range(max_iterations):
            checkpoint = None if self.is_subproblem else os.environ.get('RETARGET_TRAJECTORY_CHECKPOINT')
            if checkpoint:
                with open(checkpoint+'.tmp', 'wb') as stream:
                    np.savez(stream, q=self.q, initialization=self.initialization_q,
                             frames=np.asarray(self.frames), history=json.dumps(self.history))
                os.replace(checkpoint+'.tmp', checkpoint)
            nominal = self.linearize()
            n, m = len(nominal.q), len(nominal.h)
            certificates = np.asarray(self._frames('certificate', np.concatenate(
                (self.q, nominal.q.reshape(len(self.q), -1)), axis=1)))
            frame_errors = certificates[:, 0]
            stationarity = float(frame_errors.max())
            complementarity = float(certificates[:, 1].max())
            worst_frame = int(np.argmax(frame_errors))
            record = dict(iteration=iteration, objective=float(value), violation=violation,
                stationarity=stationarity, complementarity=complementarity, radius=radius,
                stationarity_normalization='per_frame', complementarity_normalization='absolute',
                worst_stationarity_frame=int(self.frames[worst_frame]),
                global_stationarity=float(certificates[:, 2].max()/max(1., np.max(abs(nominal.q)))),
                multiplier_certificate='current_point_nonnegative_least_squares')
            self.history.append(record)
            if not self.is_subproblem:
                print(record, flush=True)
            if stationarity <= tolerance and complementarity <= tolerance and violation <= 1e-8:
                return self.q, True
            # Static and adaptive step boxes are optimizer constraints,
            # never physical forces in the current-pose KKT certificate.
            bounds = np.tile(np.minimum(static_step, radius), len(self.q))
            problem = qpsolvers.Problem(P=nominal.P, q=nominal.q,
                G=sp.vstack([nominal.G, sp.eye(n), -sp.eye(n)], format='csc'),
                h=np.r_[nominal.h, bounds, bounds], A=nominal.A, b=nominal.b)
            try:
                solution = self.qp(problem)
            except RuntimeError:
                diagnostic = os.environ.get('RETARGET_TRAJECTORY_CHECKPOINT')
                if diagnostic:
                    import pickle
                    with open(diagnostic+'.failed-'+str(self.frames[0])+'.pkl', 'wb') as stream:
                        pickle.dump(dict(problem=problem, q=self.q, frames=self.frames,
                            history=self.history, fixed_frames=self.fixed_frames), stream)
                raise
            delta = solution.x
            # A geometric defect is the maximum over its support features.
            # Summing their nonnegative multipliers gives a valid exact-penalty
            # bound without counting the same physical penetration repeatedly.
            dual_sums = np.array([np.sum(np.maximum(solution.z[a:b], 0.))
                for a, b in zip(self.constraint_offsets[:-1], self.constraint_offsets[1:])])
            penalties = np.maximum(penalties, 2.*dual_sums)
            prediction = -float(nominal.q @ delta + .5*delta @ nominal.P @ delta) + float(penalties @ violations)
            trial = self.integrate(delta)
            new_value, new_violation = self.evaluate(trial)
            new_violations = np.asarray([row[1] for row in self._frames("evaluate", trial)])
            objective_change = self.objective_difference(self.q, trial)
            reduction = -objective_change + float(penalties @ (violations-new_violations))
            # The SOC is a normal correction of the same SQP trial. Its merit
            # reduction is tested against the original QP model, as in a
            # Maratos correction, rather than changing the predicted benefit.
            if reduction < .25*prediction and new_violation > violation:
                trial = self.correct(trial, np.max(abs(delta)))
                new_value, new_violation = self.evaluate(trial)
                new_violations = np.asarray([row[1] for row in self._frames("evaluate", trial)])
                objective_change = self.objective_difference(self.q, trial)
                reduction = -objective_change + float(penalties @ (violations-new_violations))
            ratio = reduction/prediction if prediction > 0. else -np.inf
            accepted = ratio >= 1e-4
            record.update(prediction=prediction, objective_change=objective_change,
                merit_reduction=float(reduction), max_penalty=float(penalties.max()),
                ratio=float(ratio), accepted=bool(accepted))
            if ratio < .25:
                radius *= .25
            elif ratio > .75 and np.max(abs(delta)) >= .9*radius:
                radius *= 2.
            if accepted:
                self.q, value, violation = trial, new_value, new_violation
                violations = new_violations
        return self.q, False


def current_point_multipliers(gradient, inequalities, slack, equalities):
    """Fit KKT multipliers at this pose, rather than at the QP trial pose.

    Nonnegative least squares minimizes normalized stationarity together with
    absolute complementarity. Equality multipliers are eliminated exactly.
    The returned residuals are checked against the original two tolerances;
    the least-squares objective itself is never a convergence criterion.
    Only physical inequalities belong here, not optimizer step bounds.
    """
    from scipy.optimize import nnls
    g, G, h, A = map(np.asarray, (gradient, inequalities, slack, equalities))
    projection = np.eye(len(g))
    if len(A):
        projection -= A.T @ np.linalg.solve(A @ A.T, A)
    scale = max(1., float(np.max(abs(g))))
    if not len(h):
        return projection @ g, np.zeros(0)
    matrix = np.vstack((projection @ G.T / scale, np.diag(h)))
    rhs = np.r_[-projection @ g / scale, np.zeros(len(h))]
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0.] = 1.
    dual, _ = nnls(matrix/norms, rhs, maxiter=10*len(h))
    dual /= norms
    return projection @ (g + G.T @ dual), dual


def equality_tangent_gradient(gradient, equalities):
    """Minimize the current Lagrangian gradient over equality multipliers.

    QP multipliers cancel the gradient at the QP trial, not at the current
    nonlinear iterate. Equality multipliers are unrestricted, so eliminate
    their row-space component exactly before certifying stationarity. This
    also handles fixed Schwarz halos without counting artificial forces.
    The equality rows must be linearly independent.
    """
    if equalities is None or equalities.shape[0] == 0:
        return np.asarray(gradient).copy()
    from scipy.sparse.linalg import spsolve
    A = sp.csc_matrix(equalities)
    multipliers = -spsolve((A @ A.T).tocsc(), A @ gradient)
    if not np.isfinite(multipliers).all():
        raise ValueError('Equality constraints are not linearly independent')
    return np.asarray(gradient + A.T @ multipliers)


def kkt_certificate(gradient, objective_gradient, dual, slack, nv):
    """No frame may hide its residual behind another frame's large gradient."""
    rows = np.asarray(gradient).reshape(-1, nv)
    scales = np.maximum(1., np.max(abs(np.asarray(objective_gradient).reshape(-1, nv)), axis=1))
    frame_errors = np.max(abs(rows), axis=1)/scales
    complementarity = float(np.max(abs(dual*slack), initial=0.))
    return frame_errors, complementarity


_FRAME_TRAJECTORY = None


def _initialize_frame_worker(trajectory):
    global _FRAME_TRAJECTORY
    _FRAME_TRAJECTORY = trajectory
    _FRAME_TRAJECTORY._pool = None


def _frame_batch(job):
    kind, start, poses, max_norm = job
    method = getattr(_FRAME_TRAJECTORY, '_frame_'+kind)
    return [method(start+i, q, max_norm) if kind == 'correct' else method(start+i, q)
            for i, q in enumerate(poses)]


def _solve_block(parent, job):
    start,end,left,right,poses,max_iterations,tolerance = job
    t = copy.copy(parent)
    t.q = poses.copy()
    t.initialization_q = poses.copy()
    t.frames = parent.frames[left:right]
    t.tasks = parent.tasks[left:right]
    t.references = parent.references[left:right]
    t.V = velocity_operator(len(poses),t.r.model.nv,t.dofs,t.r.time_step,
                            t.velocity_scale*t.r.time_step)
    t.Hv = (t.V.T@t.V).tocsc()
    t.history=[];t._evaluation_cache={};t._linearization_cache={}
    t._pool=None;t.frame_workers=1;t.is_subproblem=True
    t.fixed_frames=set()
    if left < start:t.fixed_frames.add(0)
    if right > end:t.fixed_frames.add(len(poses)-1)
    before,violation=t.evaluate(t.q)
    result,ok=t._solve(max_iterations,tolerance)
    after,final_violation=t.evaluate(result)
    report={'start':start,'end':end,'converged':bool(ok),
        'iterations':len(t.history),'objective_before':before,'objective_after':after,
        'violation_before':violation,'violation_after':final_violation,
        'stationarity':t.history[-1]['stationarity']}
    print({'schwarz_block':report},flush=True)
    return start,end,result[start-left:end-left].copy(),report


def _block_worker(job):
    return _solve_block(_FRAME_TRAJECTORY,job)
