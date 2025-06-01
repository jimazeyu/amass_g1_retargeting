# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import yaml
import joblib
import tqdm

"""
This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
  - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
  - target_motion_path: path to save the retargeted motion to
  - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
  - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
  - joint_mapping: mapping of joint names from source to target
  - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
  - scale: scale offset from source to target skeleton
"""

VISUALIZE = False

def main():

    # Creating a dictionary of the yaml config file
    motion_data = {"motions": {"root": "../g1_retarget_npy/"}}

    # load and visualize source skeletons
    source_sktree = SkeletonTree.from_mjcf("mjcf/smpl/smpl_humanoid_1.xml")
    source_tpose = SkeletonState.zero_pose(source_sktree)

    if VISUALIZE:
        print("source_tpose: ", source_tpose)
        plot_skeleton_state(source_tpose)

    # load and visualize target skeletons
    target_sktree = SkeletonTree.from_mjcf("mjcf/g1/g1.xml")
    target_tpose = SkeletonState.zero_pose(target_sktree)

    # adjust pose into a T Pose
    local_rotation = target_tpose.local_rotation
    local_rotation[target_sktree.index("left_shoulder_pitch_link")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[target_sktree.index("left_shoulder_pitch_link")]
    )
    local_rotation[target_sktree.index("right_shoulder_pitch_link")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
        local_rotation[target_sktree.index("right_shoulder_pitch_link")]
    )
    local_rotation[target_sktree.index("left_elbow_link")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
        local_rotation[target_sktree.index("left_elbow_link")]
    )
    local_rotation[target_sktree.index("right_elbow_link")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
        local_rotation[target_sktree.index("right_elbow_link")]
    )
    # local_rotation[target_sktree.index("left_hip_pitch_link")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([-15.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
    #     local_rotation[target_sktree.index("left_hip_pitch_link")]
    # )
    # local_rotation[target_sktree.index("right_hip_pitch_link")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([-15.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
    #     local_rotation[target_sktree.index("right_hip_pitch_link")]
    # )
    # local_rotation[target_sktree.index("left_knee_link")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([30.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
    #     local_rotation[target_sktree.index("left_knee_link")]
    # )
    # local_rotation[target_sktree.index("right_knee_link")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([30.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
    #     local_rotation[target_sktree.index("right_knee_link")]
    # )
    translation = target_tpose.root_translation
    translation += torch.tensor([0, 0, 0.7])

    if VISUALIZE:
        print("target_tpose: ", target_tpose)
        plot_skeleton_state(target_tpose)

    # load and visualize source motion sequence
    motions_skeleton_dict = joblib.load("dataset/amass/pkls/amass_isaac_im_train_take6_upright_slim_skeleton.pkl")


    # retarget each motion
    for motion_name in tqdm.tqdm(motions_skeleton_dict.keys()):
        # motion_name = np.random.choice(list(motions_skeleton_dict.keys()))
        source_motion = motions_skeleton_dict[motion_name]

        if VISUALIZE:
            print("source_motion: ", source_motion)
            print("dict: ", source_motion.to_dict().keys())
            plot_skeleton_motion_interactive(source_motion)


        # parse data from retarget config
        joint_mapping = {
            "Pelvis": "pelvis",
            "L_Hip": "left_hip_pitch_link",
            "L_Knee": "left_knee_link",
            "L_Ankle": "left_ankle_pitch_link",
            "R_Hip": "right_hip_pitch_link",
            "R_Knee": "right_knee_link",
            "R_Ankle": "right_ankle_pitch_link",
            "Torso": "waist_yaw_link",
            "L_Shoulder": "left_shoulder_pitch_link",
            "L_Elbow": "left_elbow_link",
            "L_Wrist": "left_wrist_roll_link",
            "R_Shoulder": "right_shoulder_pitch_link",
            "R_Elbow": "right_elbow_link",
            "R_Wrist": "right_wrist_roll_link"
        }
        rotation_to_target_skeleton = torch.tensor([0.0, 0.0, 0.0, 1.0])
        # print("run retargeting")
        # run retargeting
        target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=joint_mapping,
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=1.0
        )
        # print("target_motion: ", target_motion)
        # plot_skeleton_motion_interactive(target_motion)

        # keep frames between [trim_frame_beg, trim_frame_end - 1]
        frame_beg = -1
        frame_end = -1
        if (frame_beg == -1):
            frame_beg = 0
            
        if (frame_end == -1):
            frame_end = target_motion.local_rotation.shape[0]
            
        local_rotation = target_motion.local_rotation
        root_translation = target_motion.root_translation
        local_rotation = local_rotation[frame_beg:frame_end, ...]
        root_translation = root_translation[frame_beg:frame_end, ...]
        
        # new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
        # target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

        # # need to convert some joints from 3D to 1D (e.g. elbows and knees)
        # target_motion = project_joints(target_motion)

        # move the root so that the feet are on the ground
        # local_rotation = target_motion.local_rotation
        # root_translation = target_motion.root_translation
        tar_global_pos = target_motion.global_translation[frame_beg:frame_end, ...]
        min_h = torch.min(tar_global_pos[..., 2])
        root_translation[:, 2] += -min_h
        # hardcodded root height offset
        root_translation[:, 2] += 0.15

        # adjust the height of the root to avoid ground penetration
        # root_height_offset = retarget_data["root_height_offset"]
        # root_translation[:, 2] += root_height_offset
        
        new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
        target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

        # save retargeted motion
        target_motion.to_file("g1_retarget_npy/" + motion_name + ".npy")

        # visualize retargeted motion
        # plot_skeleton_motion_interactive(target_motion)

        motion_data["motions"][motion_name] = {
            "description": motion_name,
            "difficulty": 1,
            "trim_beg": -1,
            "trim_end": -1,
            "weight": 1.0
        }

    with open('motions_amass_autogen.yaml', 'w') as file:
        yaml.dump(motion_data, file)

if __name__ == '__main__':
    main()