from __future__ import absolute_import, division

import numpy as np
from utils.camera import normalize_screen_coordinates, world_to_camera, wrap
from utils.quaternion import qinverse, qrot

def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # pos_3d[:, :] -= pos_3d[:, :1]  # keep this, remove at model training.
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def _restore_head_top(poses):
    """Put slot 0 back where head_top actually is: the origin of its own frame.

    data/prepare_data_mpi_inf_3dhp.py:489 subtracts slot 0 from slots 1..16 and
    leaves slot 0 itself alone ("keep trajectory in first position"), so the
    stored pose is 16 head_top-relative body joints plus one absolute
    camera-frame position. In that frame head_top sits at exactly 0, which is
    what this writes back -- an exact reconstruction, not an estimate.

    Written in place. positions_3d aliases positions in the caller, so the two
    were never independent, and zeroing twice is a no-op.
    """
    poses[:, 0] = 0
    return poses


def read_3d_data_3dhp(dataset, restore_head_top=False):
    """Expose the stored 3DHP poses as positions_3d.

    restore_head_top turns slot 0 from the camera-frame trajectory into the real
    head_top joint. Leave it off to reproduce every result recorded before this
    flag existed: the trajectory carries ~3.3 m of camera distance, which is 40%
    of the position MSE and 11.7% of the reported MPJPE, so the two settings are
    not comparable.
    """
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions = anim['positions']
            if restore_head_top:
                if isinstance(positions, (list, tuple)):
                    positions = [_restore_head_top(p) for p in positions]
                else:
                    positions = _restore_head_top(positions)
            anim['positions_3d'] = positions
    return dataset


def fetch_h36m(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_cam = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
                    # print(dataset[subject][action]['cameras'][i])
                    # return
                    cam = dataset[subject][action]['cameras'][i]['intrinsic']
                    out_cam.append([cam] * poses_3d[i].shape[0])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions, out_cam

def drop_extreme_2d(poses_3d, poses_2d, actions, cams, max_abs):
    """Drop frames whose 2D annotation is a projection blow-up, not an observation.

    3DHP's 2D keypoints are the mocap 3D projected into each camera, so when the
    global trajectory diverges the perspective divide X/Z is written to file
    unchecked. 308 training frames come out that way: the trajectory drifts
    through the principal plane over 12 contiguous bursts (absolute depth median
    0.42 m against a normal 3.39 m, minimum -0.02 m) and max|2d| reaches 23364
    where the screen itself is normalised to [-1, 1]. No camera can observe a
    point at Z = 0, so these frames are invalid annotations rather than hard
    examples.

    The local pose survives -- bone lengths stay within 0.975-1.044 of normal --
    so only the 2D input is affected, and with --restore_head_top the 3D target
    is clean too. But the input still poisons BatchNorm: train-mode BN folds the
    outlier into the batch statistics, which momentum then writes into
    running_mean/var. Those decay over ~10 batches, so an outlier landing near
    the end of an epoch reaches eval intact and valid MPJPE jumps 3-9x.

    Filter the training split only. The evaluation split has to stay fixed for
    MPJPE to remain comparable across runs, and at the default threshold it
    contains nothing to drop anyway (TS1-TS4 peak at 5.18).
    """
    kept_3d = None if poses_3d is None else []
    kept_2d, kept_actions, kept_cams = [], [], []
    dropped = 0

    for i, seq_2d in enumerate(poses_2d):
        seq_2d = np.asarray(seq_2d)
        keep = np.abs(seq_2d).reshape(len(seq_2d), -1).max(1) <= max_abs
        dropped += int((~keep).sum())
        if not keep.any():
            continue
        kept_2d.append(seq_2d[keep])
        kept_actions.append([a for a, k in zip(actions[i], keep) if k])
        if kept_3d is not None:
            kept_3d.append(np.asarray(poses_3d[i])[keep])
        if cams:
            kept_cams.append([c for c, k in zip(cams[i], keep) if k])

    return kept_3d, kept_2d, kept_actions, kept_cams, dropped
