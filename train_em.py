import argparse
from os import path
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import time

from data.prepare_data_h3wb import Human3WBDataset, Human3WBTestDataset
from data_loader import PoseBuffer, PoseDataSet
from models.linear_model import LinearModel
from models.loss_net import LinearLossNet, MarginBasedLoss
from models.sem_gcn import SemGCN, adj_mx_from_skeleton
from utils.camera import normalize_screen_coordinates
from utils.data_utils import create_2d_data, fetch_h36m, read_3d_data
from utils.device import get_device
from utils.h36m_dataset import Human36mDataset

from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.utils import AverageMeter
import wandb


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
        pin_memory=False,
    )
    return dataloader


def prepare_data_h3wb(args, train_data_path, eval_data_path):
    print("==> Preparing data...")

    dataset = Human3WBDataset(train_data_path)
    dataset_test = Human3WBTestDataset(eval_data_path, train_data_path)

    train_loader = get_dataloader(args, dataset)
    valid_loader = get_dataloader(args, dataset_test, test=True)
    return train_loader, valid_loader, dataset


def prepare_data_h36m(args, data_path):
    subjects_train = ["S1", "S5", "S6", "S7", "S8"]
    subjects_test = ["S9", "S11"]
    print("==> Preparing data...")
    h36m_dataset = Human36mDataset(data_path)
    dataset = read_3d_data(h36m_dataset)
    print("==> Loading 2D detections...")
    stride = 1
    action_filter = None

    keypoints = create_2d_data(f"data/data_2d_h36m_{args.keypoints}.npz", dataset)
    poses_train, poses_train_2d, actions_train, cams_train = fetch_h36m(
        subjects_train, dataset, keypoints, action_filter, stride
    )
    poses_valid, poses_valid_2d, actions_valid, cams_valid = fetch_h36m(
        subjects_test, dataset, keypoints, action_filter, stride
    )
    train_loader = DataLoader(
        PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    valid_loader = DataLoader(
        PoseDataSet(poses_valid, poses_valid_2d, actions_valid, cams_valid),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return train_loader, valid_loader, dataset


def main(args):
    if args.dataset == "h3wb":
        train_loader, valid_loader, dataset = prepare_data_h3wb(
            args, "data/h3wb_train.npz", "data/task1_test_3d.npz"
        )
    elif args.dataset == "h36m":
        train_loader, valid_loader, dataset = prepare_data_h36m(
            args, "data/data_3d_h36m.npz"
        )
    else:
        print(f"Invalid dataset: {args.dataset}")
        raise ValueError(f"Invalid dataset: {args.dataset}")
    is_h3wb = args.dataset == "h3wb"

    print("==> Data loaded...")
    device = get_device()
    if args.pos_loss == "mse":
        criterion = nn.MSELoss(reduction="mean").to(device)
    elif args.pos_loss == "mpjpe":
        criterion = mpjpe
    num_joints = dataset.skeleton().num_joints()
    print(f"==> Number of joints: {num_joints}")

    if args.task_net == "linear":
        model_pos = LinearModel(
            num_joints * 2,
            (num_joints - 1) * 3,
            linear_size=args.task_linear_size,
            num_stage=args.task_num_stage,
            p_dropout=args.task_dropout,
            num_joints=num_joints,
            batch_norm=args.task_batch_norm,
        )
    elif args.task_net == "linear-large":
        model_pos = LinearModel(
            num_joints * 2,
            (num_joints - 1) * 3,
            linear_size=2048,
            num_stage=3,
            p_dropout=0.5,
            num_joints=num_joints,
            batch_norm=1,
        )
    elif args.task_net == "semgcn":
        model_pos = SemGCN(
            adj=adj_mx_from_skeleton(dataset.skeleton()),
            hid_dim=128,
            num_layers=4,
            num_joints=num_joints,
            # p_dropout=None if args.task_dropout == 0.0 else args.task_dropout,
            # nodes_group=dataset.skeleton().joints_group() if args.non_local else None,
        )

    if args.checkpoint is not None:
        try:
            model_pos.load_state_dict(torch.load(args.checkpoint, map_location=device))
        except:
            try:
                model_pos = torch.load(args.checkpoint, map_location=device)
            except:
                print(f"Invalid checkpoint: {args.checkpoint}")
                raise ValueError(f"Invalid checkpoint: {args.checkpoint}")

    print(
        f"==> Number of parameters: {sum(p.numel() for p in model_pos.parameters()):,}"
    )

    model_pos.to(device)

    if args.type != "baseline":
        model_loss = LinearLossNet(
            linear_size=args.loss_linear_size,
            num_stage=args.loss_num_stage,
            p_dropout=args.loss_dropout,
            num_joints=num_joints,
            batch_norm=args.loss_batch_norm,
        )
        print(
            f"==> Number of parameters (loss-net): {sum(p.numel() for p in model_loss.parameters()):,}"
        )
        model_loss.to(device)
        model_loss.train()
        optimizer_loss = torch.optim.Adam(model_loss.parameters(), lr=args.lr_loss)
        criterion_loss = MarginBasedLoss(
            margin_ratio=args.margin_ratio, loss_type=args.em_loss
        )

    start_time = time.time()

    if args.save_path is not None:
        run_id = args.save_path.split("/")[-1]
        save_dir = args.save_path.replace(run_id, "")
        os.makedirs(path.dirname(save_dir), exist_ok=True)
    else:
        run_id = None
        save_dir = None

    run = (
        wandb.init(
            project="pose-lift",
            config=vars(args),
            # resume="allow" if args.checkpoint is not None else None,
            id=run_id,
            tags=[args.wandb_tag] if args.wandb_tag is not None else None,
        )
        if not args.no_logging
        else None
    )

    epoch = 1
    epochs = args.num_epoch if args.patience < 0 else -1
    model_pos.eval()

    while True:
        # for epoch in range(1, epochs + 1):
        epoch_loss_energy = AverageMeter()
        epoch_e_diff = AverageMeter()
        epoch_mpjpe = AverageMeter()
        epoch_e_diffs = []

        for i, batch in enumerate(tqdm(train_loader)):
            model_loss.train()
            targets_3d, inputs_2d = batch[0].to(device), batch[1].to(device)
            batch_size = targets_3d.size(0)
            targets_3d = (
                targets_3d[:, :, :] - targets_3d[:, :1, :]
            )  # the output is relative to the 0 joint
            if args.type == "gaussian":
                # average_3d = targets_3d.mean(dim=1, keepdim=True)
                # print(average_3d)
                # outputs_3d = targets_3d + targets_3d * torch.randn_like(targets_3d)
                outputs_3d = targets_3d + torch.randn_like(targets_3d) * 8
                energy_hat = model_loss(inputs_2d, outputs_3d)
                energy_label = model_loss(inputs_2d, targets_3d)
                e_diff = energy_hat - energy_label
                epoch_e_diff.update(e_diff.mean().item(), batch_size)
                optimizer_loss.zero_grad()
                loss_loss_net = criterion_loss(
                    outputs_3d, targets_3d, energy_hat, energy_label
                )
                epoch_loss_energy.update(loss_loss_net.item(), batch_size)
                loss_loss_net.backward()
                optimizer_loss.step()
                mpjpe = criterion(outputs_3d, targets_3d)
                epoch_mpjpe.update(mpjpe.item(), batch_size)
                epoch_e_diffs.append(e_diff.mean().item())
            elif args.type == "tasknet":
                with torch.no_grad():
                    outputs_3d = model_pos(inputs_2d)
                energy_hat = model_loss(inputs_2d, outputs_3d)
                energy_label = model_loss(inputs_2d, targets_3d)
                e_diff = energy_hat - energy_label
                epoch_e_diff.update(e_diff.mean().item(), batch_size)
                optimizer_loss.zero_grad()
                loss_loss_net = criterion_loss(
                    outputs_3d, targets_3d, energy_hat, energy_label
                )
                loss_loss_net.backward()
                optimizer_loss.step()
                mpjpe = criterion(outputs_3d, targets_3d)
                epoch_mpjpe.update(mpjpe.item(), batch_size)
                epoch_e_diffs.append(e_diff.mean().item())

            else:
                print(f"Invalid type: {args.type}")
                raise ValueError(f"Invalid type: {args.type}")

        if epoch % args.eval_interval == 0:
            epoch_loss_energy_eval = AverageMeter()
            epoch_e_diff_eval = AverageMeter()
            epoch_e_diffs_eval = []
            model_loss.eval()
            for i, batch in enumerate(valid_loader):
                targets_3d, inputs_2d = batch[0].to(device), batch[1].to(device)
                batch_size = targets_3d.size(0)
                targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]
                if args.type == "gaussian":
                    outputs_3d = targets_3d + torch.randn_like(targets_3d) * 0.1
                elif args.type == "tasknet":
                    with torch.no_grad():
                        outputs_3d = model_pos(inputs_2d)

                # outputs_3d = targets_3d + torch.randn_like(targets_3d) * 3
                # with torch.no_grad():
                #     outputs_3d = model_pos(inputs_2d)

                energy_hat = model_loss(inputs_2d, outputs_3d)
                energy_label = model_loss(inputs_2d, targets_3d)
                e_diff = energy_hat - energy_label
                epoch_e_diff_eval.update(e_diff.mean().item(), batch_size)
                epoch_e_diffs_eval.append(e_diff.mean().item())
                loss_loss_net = criterion_loss(
                    targets_3d, targets_3d, energy_hat, energy_label
                )
                epoch_loss_energy_eval.update(loss_loss_net.item(), batch_size)

            print(
                f"==> Epoch {epoch}: train loss: {epoch_loss_energy.avg:.2E}, e_diff: {epoch_e_diff.avg:.2E}, ratio:{np.mean(epoch_e_diffs)/np.std(epoch_e_diffs):.2f}, mpjpe: {epoch_mpjpe.avg:.2f}"
            )
            print(
                f"    Epoch {epoch}: eval loss: {epoch_loss_energy_eval.avg:.2E}, e_diff: {epoch_e_diff_eval.avg:.2E}, ratio:{np.mean(epoch_e_diffs_eval)/np.std(epoch_e_diffs_eval):.2f}, time: {time.time() - start_time:.2f}s"
            )
            start_time = time.time()

        if epoch == epochs:
            break
        epoch += 1

    if save_dir is not None:
        torch.save(
            model_loss,
            args.save_path,
        )
    elif run is not None:
        torch.save(
            model_loss,
            f"temp_ckpts/{run.name}-final.pth",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_loss", type=float, default=1e-3)
    parser.add_argument("--energy_weight", type=float, default=1e-3)
    parser.add_argument("--margin_ratio", type=float, default=1)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=500)
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--type", type=str, default="baseline")
    parser.add_argument("--no_logging", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
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
    parser.add_argument("--loss_batch_norm", type=int, default=0)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--task_eval_during_dynamic", type=int, default=0)
    parser.add_argument("--loss_eval_during_dynamic", type=int, default=1)
    parser.add_argument("--eval_by_pa", type=int, default=0)
    parser.add_argument("--pos_loss", type=str, default="mse")
    parser.add_argument("--em_loss", type=str, default="mse")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--wandb_tag", type=str, default=None)
    # parser.add_argument("--non_local", type=int, default=0)

    args = parser.parse_args()
    print(args)

    random_seed = args.seed
    set_seed(random_seed)

    main(args)
