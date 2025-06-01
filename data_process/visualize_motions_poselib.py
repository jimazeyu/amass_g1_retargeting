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


if __name__ == "__main__":
    motions_skeleton_dict = joblib.load("dataset/amass/pkls/amass_isaac_im_train_take6_upright_slim_skeleton.pkl")
    
    # random sample a motion
    motion_name = np.random.choice(list(motions_skeleton_dict.keys()))
    print(f"motion_name: {motion_name}")

    # print skeleton node names
    tpose = SkeletonTree.from_mjcf("mjcf/smpl/smpl_humanoid_1.xml")
    print("tpose: ", tpose.node_names)

    motion_skeleton = motions_skeleton_dict[motion_name]
    plot_skeleton_motion_interactive(motion_skeleton)