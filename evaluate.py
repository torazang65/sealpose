import argparse
import numpy as np
import torch

from data.prepare_data_h3wb import Human3WBDataset, Human3WBTestDataset
from data_loader import PoseBuffer, PoseDataSet
from eval import evaluate
from models.linear_model import LinearModel
from utils.camera import normalize_screen_coordinates
from utils.data_utils import create_2d_data, fetch_h36m, read_3d_data
from utils.device import get_device
from utils.h36m_dataset import Human36mDataset

from torch.utils.data import DataLoader

from utils.seed import set_seed


def fetch(
    subjects,
    keypoints,  # New
    dataset,  # New
    stride,  # New
    action_filter=None,
    subset=1,
    parse_3d_poses=True,
):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), "Camera count mismatch"
                for cam in cams:
                    if "intrinsic" in cam:
                        out_camera_params.append(cam["intrinsic"])

            if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                poses_3d = dataset[subject][action]["positions_3d"]
                assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    # >> Moved to function arguments <<
    # stride = args.experiment.downsample
    # if subset < 1:
    #     for i in range(len(out_poses_2d)):
    #         n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
    #         start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
    #         out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
    #         if out_poses_3d is not None:
    #             out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


def get_dataloader(args, dataset, subjects=None, test=False):
    subjects = dataset.subjects() if subjects is None else subjects
    print("Preparing 3D data...")
    for subject in subjects:
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if "positions" in anim:
                positions_3d = []

                for ind, cam in enumerate(anim["positions_3d"]):
                    pos_3d = anim["positions_3d"][ind]
                    pos_3d = pos_3d / 1000.0  # lets divide by 1000 to convert meters
                    positions_3d.append(pos_3d)
                anim["positions_3d"] = positions_3d

    print("Preparing 2D detections...")
    ################### 2D data preparation
    keypoints = {}
    for subject in subjects:
        keypoints[subject] = {}
        for action in dataset[subject].keys():
            keypoints[subject][action] = []
            for cam_idx, kps in enumerate(dataset[subject][action]["pose_2d"]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam["res_w"], h=cam["res_h"]
                )
                keypoints[subject][action].append(kps)

    stride = 1
    action_filter = None

    # >> Modification <<
    _cameras_train, poses_3d, poses_2d = fetch(
        subjects=subjects,
        keypoints=keypoints,
        dataset=dataset,
        stride=stride,
        action_filter=action_filter,
    )
    dataloader = DataLoader(
        PoseBuffer(poses_3d, poses_2d),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader


def prepare_data_h3wb_val(args, train_data_path, eval_data_path):
    # metadata = np.load(training_data_path, allow_pickle=True)['metadata'].item() # todo
    dataset = Human3WBDataset(train_data_path)
    dataset_test = Human3WBTestDataset(eval_data_path, train_data_path)

    # train_loader = get_dataloader(dataset, subjects=["S1", "S5", "S6"])
    # valid_loader = get_dataloader(dataset, subjects=["S7"])

    valid_loader = get_dataloader(args, dataset_test, test=True)
    return valid_loader, dataset


def prepare_data_h36m_val(args, data_path):
    subjects_test = ["S9", "S11"]
    h36m_dataset = Human36mDataset(data_path)
    dataset = read_3d_data(h36m_dataset)
    stride = 1
    action_filter = None

    keypoints = create_2d_data(f"data/data_2d_h36m_{args.keypoints}.npz", dataset)
    poses_valid, poses_valid_2d, actions_valid, cams_valid = fetch_h36m(
        subjects_test, dataset, keypoints, action_filter, stride
    )
    valid_loader = DataLoader(
        PoseDataSet(poses_valid, poses_valid_2d, actions_valid, cams_valid),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return valid_loader, dataset


def main(args):
    if args.dataset == "h3wb":
        if args.incomplete:
            valid_loader, dataset = prepare_data_h3wb_val(
                args, "data/h3wb_train.npz", "data/task2_test_3d.npz"
            )
        else:
            valid_loader, dataset = prepare_data_h3wb_val(
                args, "data/h3wb_train.npz", "data/task1_test_3d.npz"
            )
    elif args.dataset == "h36m":
        valid_loader, dataset = prepare_data_h36m_val(args, "data/data_3d_h36m.npz")
    elif args.dataset == "3dhp":
        mpi3d_npz = np.load("data_extra/test_set/test_3dhp.npz")
        tmp = mpi3d_npz
        valid_loader = DataLoader(
            PoseBuffer([tmp["pose3d"]], [tmp["pose2d"]]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    is_h3wb = args.dataset == "h3wb"

    device = get_device()
    num_joints = 133 if is_h3wb else 16

    model_pos = LinearModel(
        num_joints * 2,
        (num_joints - 1) * 3,
        linear_size=args.task_linear_size,
        num_stage=args.task_num_stage,
        p_dropout=args.task_dropout,
        num_joints=num_joints,
        batch_norm=args.task_batch_norm,
    )
    try:
        model_pos.load_state_dict(torch.load(args.checkpoint), map_location=device)
    except:
        model_pos = torch.load(args.checkpoint, map_location=device)

    model_pos.to(device)

    error_h3wb_p1, error_h3wb_p2, pelvis_aligned_mpjpe = evaluate(
        valid_loader, model_pos, device, flipaug=args.flip, is_h3wb=is_h3wb
    )
    print(f"==> {error_h3wb_p1:.3f}, {error_h3wb_p2:.3f}")
    # print(f"==> MPJPE: {error_h3wb_p1:.2f}mm, P-MPJPE: {error_h3wb_p2:.2f}mm")
    if is_h3wb:  # pelvis_aligned_mpjpe is tuple of floats
        # print(f"==> Pelvis-aligned MPJPE: {pelvis_aligned_mpjpe[0]:.2f}mm, {pelvis_aligned_mpjpe[1]:.2f}mm")
        print(
            f"==> {pelvis_aligned_mpjpe[0]:.3f}, {pelvis_aligned_mpjpe[1]:.3f}, {pelvis_aligned_mpjpe[2]:.3f}, {pelvis_aligned_mpjpe[3]:.3f}, {pelvis_aligned_mpjpe[4]:.3f}, {pelvis_aligned_mpjpe[5]:.3f}"
        )

    print("Done\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--type", type=str, default="baseline")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="h3wb")
    parser.add_argument("--task_net", type=str, default="linear")
    parser.add_argument("--task_dropout", type=float, default=0.5)
    parser.add_argument("--task_num_stage", type=int, default=2)
    parser.add_argument("--task_linear_size", type=int, default=1024)
    parser.add_argument("--task_batch_norm", type=int, default=1)
    parser.add_argument("--keypoints", type=str, default="gt")
    parser.add_argument("--loss_net", type=str, default="linear")
    parser.add_argument("--loss_dropout", type=float, default=0.5)
    parser.add_argument("--loss_num_stage", type=int, default=2)
    parser.add_argument("--loss_linear_size", type=int, default=1024)
    parser.add_argument("--loss_batch_norm", type=int, default=1)
    parser.add_argument("--flip", type=int, default=0)
    parser.add_argument("--incomplete", type=int, default=0)

    args = parser.parse_args()
    print(
        f"Evaluate {args.type.upper()} model on {args.dataset.upper()} with 2D {args.keypoints.upper()} keypoints"
    )

    random_seed = args.seed
    set_seed(random_seed)

    main(args)
