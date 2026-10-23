"""Strict, resumable Bones G1 batch retarget; publish only complete feasible clips."""
from __future__ import annotations
import argparse, concurrent.futures, contextlib, csv, gzip, hashlib, json, os
from pathlib import Path
import sys, tempfile, time, traceback
from functools import lru_cache
from importlib.metadata import version
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import mujoco
import yaml
from source import convert_one_file
from solver import RobotRetarget
import targets
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/robot/booster_k1.yaml'
KEYS=['ik_match_table','joints_limit_offset_degrees','posture_regularization',
      'ground_nonpenetration','contact_grounding','root_trajectory_constraint',
      'self_collision_avoidance','temporal_regularization','max_ik_iterations',
      'ik_convergence_tolerance','trajectory_fitting']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024*1024): h.update(block)
    return h.hexdigest()

@lru_cache(maxsize=1)
def producer_fingerprint():
    paths=[ROOT/'requirements.lock', ROOT/'.python-version', ROOT/'validation/acceptance.json',
           *sorted((ROOT/'scripts').glob('*.py')),
           *sorted((ROOT/'config/robot').glob('*.yaml')),
           *sorted(p for p in (ROOT/'asset/robot/booster_k1').rglob('*') if p.is_file()),
           *sorted(p for p in (ROOT/'asset/robot/g1_description').rglob('*') if p.is_file())]
    identity={'files':{str(p.relative_to(ROOT)):sha256(p) for p in paths},
              'packages':{p:version(p) for p in ['numpy','scipy','mujoco','mink','qpsolvers','daqp','clarabel','cffi','setuptools']}}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

def run_clip(entry, output_root):
    output_root=Path(output_root); member=Path(entry['member'])
    if member.is_absolute() or '..' in member.parts: raise ValueError('Invalid member path')
    stem=member.with_suffix(''); destination=output_root/'motion'/member
    record=output_root/'status'/Path(str(stem)+'.json')
    record.parent.mkdir(parents=True,exist_ok=True)
    producer=producer_fingerprint()
    if record.exists():
        old=json.loads(record.read_text())
        if (old.get('status')=='accepted' and old.get('source_sha256')==entry['sha256']
                and old.get('producer_sha256')==producer
                and destination.exists() and sha256(destination)==old.get('output_sha256')):
            return {**old,'resumed':True}
    started=time.time(); result={'member':entry['member'],'source_sha256':entry['sha256'],
                                 'status':'failed','fps':120,'producer_sha256':producer}
    log_path=output_root/'logs'/Path(str(stem)+'.log');log_path.parent.mkdir(parents=True,exist_ok=True)
    r=None
    try:
        os.chdir(ROOT)
        if sha256(entry['path'])!=entry['sha256']:raise ValueError('Source checksum changed')
        source_model=mujoco.MjModel.from_xml_path(str(ROOT/'asset/robot/g1_description/mjcf/g1.xml'))
        expected=[source_model.joint(j).name+'_dof' for j in range(source_model.njnt)
                  if source_model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
        with open(entry['path'],newline='') as f:
            header=next(csv.reader(f))
        if [x for x in header if x.endswith('_dof')]!=expected:
            raise ValueError('Source joint columns/order differ from the G1 model')
        with log_path.open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), tempfile.TemporaryDirectory(dir=output_root/'work') as temp:
            temp=Path(temp);converted=temp/'g1.csv';keypoints=temp/'keypoints.pkl'
            frames,joints=convert_one_file(Path(entry['path']),converted,.01)
            if frames<2:raise ValueError('Motion requires at least two frames')
            targets.prepare_keypoints(converted, keypoints)
            c=yaml.safe_load(CONFIG.read_text())
            r=RobotRetarget(str(ROOT/c['robot_xml_path']),str(keypoints),
                contact_body_names=c['contact_links'],contact_position_cost=c['contact_pos_fixed_factor'],
                **{k:c[k] for k in KEYS})
            result['preparation_seconds'] = time.time()-started
            r.retarget()
            audit_started = time.perf_counter()
            bad=[x['frame'] for x in r.solver_diagnostics if x['reason']!='kkt_tolerance']
            result.update(expected_frames=frames,frames=len(r.result_pos),initialization_budget_frames=bad)
            if c.get('trajectory_fitting',{}).get('enabled',False):
                if not getattr(r,'trajectory_converged',False):
                    raise ValueError('Joint trajectory solve did not satisfy KKT conditions')
                result['trajectory_converged']=True
                result['final_kkt']=r.trajectory_diagnostics[-1]
                from acceptance import validate_kkt
                validate_kkt(result['final_kkt'])
            elif bad:
                raise ValueError('One or more frame solves did not satisfy KKT conditions')
            q=np.asarray(r.result_pos)
            if q.dtype != np.float64:
                raise ValueError('Solver output is not float64')
            if len(q)!=frames or not np.isfinite(q).all():raise ValueError('Incomplete/nonfinite output')
            ground=np.inf; clearance=np.inf; joint_violation=0.
            for pose in q:
                r.configuration.update(q=pose)
                ground=min(ground,float(r.whole_body_ground_limit.gaps(r.configuration).min()))
                clearance=min(clearance,r.minimum_self_collision_distance())
            for j in range(r.model.njnt):
                if r.model.jnt_limited[j]:
                    values=q[:,r.model.jnt_qposadr[j]];lo,hi=r.model.jnt_range[j]
                    joint_violation=max(joint_violation,float(np.maximum(lo-values,values-hi).max()))
            root_error=float(np.abs(q[:,:2]-r.source_root_positions[:,:2]).max())
            result.update(minimum_ground_gap_m=ground,minimum_self_clearance_m=clearance,
                          joint_limit_violation_rad=joint_violation,root_xy_error_m=root_error,
                          max_joint_speed_rad_s=float(np.abs(np.diff(q[:,7:],axis=0)).max()*120),
                          max_joint_acceleration_rad_s2=float(np.abs(np.diff(q[:,7:],n=2,axis=0)).max()*14400) if frames>2 else 0.)
            mapping=c['posture_regularization']['joint_map']
            observed=np.stack([r.source_joint_positions[:,r.source_joint_names.index(src)]*scale
                               +np.deg2rad(offset) for src,scale,offset in mapping.values()],axis=1)
            addresses=[r.model.joint(name).qposadr[0] for name in mapping]
            velocity_error=np.diff(q[:,addresses]-observed,axis=0)*r.fps
            result.update(source_mapped_max_joint_speed_rad_s=float(np.abs(np.diff(observed,axis=0)).max()*r.fps),
                          source_velocity_residual_rms_rad_s=float(np.sqrt(np.mean(velocity_error**2))))
            if ground < -1e-8 or clearance < r.self_collision_minimum_distance-1e-8 or joint_violation>1e-8 or root_error>1e-8:
                raise ValueError('Independent geometry/root validation failed')
            from acceptance import validate_record
            validate_record({**result, 'status': 'accepted'})
            result['audit_seconds'] = time.perf_counter()-audit_started
            destination.parent.mkdir(parents=True,exist_ok=True)
            staged=temp/'accepted.csv';r.save_results_as_csv(str(staged));os.replace(staged,destination)
            result.update(status='accepted',output_sha256=sha256(destination),output_path=str(destination))
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
        with log_path.open('a') as f:traceback.print_exc(file=f)
    finally:
        if r is not None:
            result.update(expected_frames=r.num_frames, frames=len(r.result_pos),
                          stage_timings=getattr(r, 'stage_timings', {}))
            diagnostic=output_root/'diagnostics'/Path(str(stem)+'.json.gz');diagnostic.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(diagnostic,'wt') as f:
                json.dump({'initialization':r.solver_diagnostics,
                           'trajectory':r.trajectory_diagnostics,
                           'schwarz':getattr(r, 'schwarz_diagnostics', [])},f)
        result['elapsed_seconds']=time.time()-started
        staged_record=record.with_suffix('.json.tmp')
        staged_record.write_text(json.dumps(result,indent=2))
        os.replace(staged_record,record)
    return result
