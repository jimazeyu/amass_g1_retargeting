from ast import Try
import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
import glob
import os
import sys
import pdb
import os.path as osp
from pathlib import Path

sys.path.append(os.getcwd())

# from smpl_sim.khrylib.utils import get_body_qposaddr
from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import argparse

def run(in_file: str, out_file: str):

    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    print(robot_cfg)

    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="dataset/smpl/smpl",
    )

    amass_data = joblib.load(in_file)

    double = False

    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    

    amass_full_motion_dict = {}
    amass_full_motion_skeleton_dict = {}
    for key_name in tqdm(amass_data.keys()):
        smpl_data_entry = amass_data[key_name]
        B = smpl_data_entry['pose_aa'].shape[0]

        start, end = 0, 0

        pose_aa = smpl_data_entry['pose_aa'].copy()[start:]
        root_trans = smpl_data_entry['trans'].copy()[start:]
        B = pose_aa.shape[0]

        beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
        if len(beta.shape) == 2:
            beta = beta[0]

        gender = smpl_data_entry.get("gender", "neutral")
        fps = 30.0

        if isinstance(gender, np.ndarray):
            gender = gender.item()

        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        if gender == "neutral":
            gender_number = [0]
        elif gender == "male":
            gender_number = [1]
        elif gender == "female":
            gender_number = [2]
        else:
            import ipdb
            ipdb.set_trace()
            raise Exception("Gender Not Supported!!")

        smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
        pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

        num = 1
        if double:
            num = 2
        for idx in range(num):
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

            gender_number, beta[:], gender = [0], 0, "neutral"
            print("using neutral model")

            smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
            smpl_local_robot.write_xml("mjcf/smpl/smpl_humanoid_1.xml")
            skeleton_tree = SkeletonTree.from_mjcf("mjcf/smpl/smpl_humanoid_1.xml")

            root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)

            if robot_cfg['upright_start']:
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
                pose_quat = new_sk_state.local_rotation.numpy()

                ############################################################
                # key_name_dump = key_name + f"_{idx}"
                key_name_dump = key_name
                if idx == 1:
                    left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
                    pose_quat_global = pose_quat_global[:, left_to_right_index]
                    pose_quat_global[..., 0] *= -1
                    pose_quat_global[..., 2] *= -1

                    root_trans_offset[..., 1] *= -1
                ############################################################

            new_motion_out = {}
            new_motion_out['pose_quat_global'] = pose_quat_global
            new_motion_out['pose_quat'] = pose_quat
            new_motion_out['trans_orig'] = root_trans
            new_motion_out['root_trans_offset'] = root_trans_offset
            new_motion_out['beta'] = beta
            new_motion_out['gender'] = gender
            new_motion_out['pose_aa'] = pose_aa
            new_motion_out['fps'] = fps

            amass_full_motion_dict[key_name_dump] = new_motion_out

            # save motions in skeleton
            sk_motion_out = SkeletonMotion.from_skeleton_state(new_sk_state, fps)
            amass_full_motion_skeleton_dict[key_name_dump] = sk_motion_out
            # plot_skeleton_motion_interactive(sk_motion_out)

    Path(out_file).parents[0].mkdir(parents=True, exist_ok=True)
    joblib.dump(amass_full_motion_dict, out_file)
    joblib.dump(amass_full_motion_skeleton_dict, out_file.replace(".pkl", "_skeleton.pkl"))
    return

# import ipdb

# ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="sample_data/amass_copycat_take6_train.pkl")
    parser.add_argument("--out_file", type=str, default="dataset/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl")
    args = parser.parse_args()
    run(
        in_file=args.in_file,
        out_file=args.out_file
    )
