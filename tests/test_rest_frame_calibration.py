from pathlib import Path
import sys
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from targets import derive_rest_frame_calibrations

def test_arm_frames_map_target_longitudinal_axes_to_source():
    root=Path(__file__).resolve().parents[1]
    calibration=derive_rest_frame_calibrations(root/'config/robot/g1.yaml',root/'config/robot/booster_k1.yaml',root/'asset/robot/booster_k1/K1_22dof.xml')
    source=mujoco.MjModel.from_xml_path(str(root/'asset/robot/g1_description/mjcf/g1.xml'));sd=mujoco.MjData(source);mujoco.mj_forward(source,sd)
    target=mujoco.MjModel.from_xml_path(str(root/'asset/robot/booster_k1/K1_22dof.xml'));td=mujoco.MjData(target);mujoco.mj_forward(target,td)
    for side in ['left','right']:
        a=source.body(side+'_elbow_link').id;b=source.body(side+'_wrist_yaw_link').id
        c=target.body(side+'_hand_link').id;d=target.body(side+'_hand_end_link').id
        sv=sd.xmat[a].reshape(3,3).T@(sd.xpos[b]-sd.xpos[a]);sv/=np.linalg.norm(sv)
        tv=td.xmat[c].reshape(3,3).T@(td.xpos[d]-td.xpos[c]);tv/=np.linalg.norm(tv)
        C=calibration[side+'_arm']
        np.testing.assert_allclose(C@tv,sv,atol=1e-12)
        np.testing.assert_allclose(C.T@C,np.eye(3),atol=1e-12)
        assert np.linalg.det(C)>0.999999
