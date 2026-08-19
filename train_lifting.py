import argparse
from os import path
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import time

from data.prepare_data_h3wb import Human3WBDataset, Human3WBTestDataset
from data.prepare_data_mpi_inf_3dhp import MpiInf3dhpDataset
from data_loader import PoseBuffer, PoseDataSet
from eval import evaluate
from models.linear_model import LinearModel
from models.loss_net import LinearLossNet, MarginBasedLoss, NCELoss
from models.sem_gcn import SemGCN, adj_mx_from_skeleton
from models.video_pose import TemporalModelOptimized1f
from utils.camera import normalize_screen_coordinates
from utils.data_utils import create_2d_data, fetch_h36m, read_3d_data, read_3d_data_3dhp
from utils.device import get_device
from utils.h36m_dataset import Human36mDataset

from torch.utils.data import DataLoader

from utils.loss import BoneDirectionLoss, mpjpe
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

def prepare_data_3dhp(args, data_path):
    subjects_train = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    subjects_test = ["TS1", "TS2", "TS3", "TS4"]
    print("==> Preparing data...")
    h36m_dataset = MpiInf3dhpDataset(data_path)
    dataset = read_3d_data_3dhp(h36m_dataset)
    print("==> Loading 2D detections...")
    stride = 1
    action_filter = None

    keypoints = create_2d_data(f"data/data_2d_mpi_inf_3dhp_{args.keypoints}.npz", dataset)
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
        raw_bs = args.batch_size
        # args.batch_size = args.loss_batch_size
        args.batch_size = args.batch_size * args.num_samples
        train_loader_loss, _, _ = prepare_data_h3wb(
            args, "data/h3wb_train.npz", "data/task1_test_3d.npz"
        )
        args.batch_size = raw_bs
    elif args.dataset == "h36m":
        train_loader, valid_loader, dataset = prepare_data_h36m(
            args, "data/data_3d_h36m.npz"
        )
        raw_bs = args.batch_size
        # args.batch_size = args.loss_batch_size
        args.batch_size = args.batch_size * args.num_samples
        train_loader_loss, _, _ = prepare_data_h36m(args, "data/data_3d_h36m.npz")
        args.batch_size = raw_bs
    elif args.dataset == "3dhp":
        train_loader, valid_loader, dataset = prepare_data_3dhp(
            args, "data/data_3d_mpi_inf_3dhp.npz"
        )
        raw_bs = args.batch_size
        # args.batch_size = args.loss_batch_size
        args.batch_size = args.batch_size * args.num_samples
        train_loader_loss, _, _ = prepare_data_3dhp(args, "data/data_3d_mpi_inf_3dhp.npz")
        args.batch_size = raw_bs
    else:
        print(f"Invalid dataset: {args.dataset}")
        raise ValueError(f"Invalid dataset: {args.dataset}")
    is_h3wb = args.dataset == "h3wb"

    def center(poses):
        """Move a batch of ground-truth poses into the model's output frame.

        The loaders hand back raw stored poses -- read_3d_data_3dhp passes
        positions through untouched and read_3d_data has its centering commented
        out with "remove at model training" -- so every ground-truth batch has to
        be centred here, not just the task net's. On 3DHP the raw frame puts the
        root 0.71 m off the origin, so a batch that misses this sits 713 mm from
        the model's output space, 13x the task net's own error.
        """
        if args.dataset == "3dhp":
            return poses - poses[:, 14:15, :]        # 14 is the pelvis
        if not is_h3wb or args.centering == "zero":
            return poses - poses[:, :1, :]
        return poses - (poses[:, 11:12, :] + poses[:, 12:13, :]) / 2

    print("==> Data loaded...")
    device = get_device()
    if args.pos_loss == "mse":
        criterion = nn.MSELoss(reduction="mean").to(device)
    elif args.pos_loss == "mpjpe":
        criterion = mpjpe
    num_joints = dataset.skeleton().num_joints()
    print(f"==> Number of joints: {num_joints}")

    # Length-weighted bone-direction term (report/check_bodyness.py, section 7).
    # 3DHP's joint 0 sits 1.2-6.3 m from the head in the ground truth, so it is
    # not a body joint and is dropped from the bone set.
    criterion_dir = None
    if args.dir_weight > 0:
        criterion_dir = BoneDirectionLoss(
            dataset.skeleton().parents(),
            exclude=[0] if args.dataset == "3dhp" else [],
        ).to(device)
        print(f"==> Bone-direction loss on {len(criterion_dir.child)} bones, "
              f"weight {args.dir_weight}")

    if args.task_net == "linear":
        model_pos = LinearModel(
            num_joints * 2,
            num_joints * 3,
            linear_size=args.task_linear_size,
            num_stage=args.task_num_stage,
            p_dropout=args.task_dropout,
            num_joints=num_joints,
            batch_norm=args.task_batch_norm,
        )
    elif args.task_net == "linear-large":
        model_pos = LinearModel(
            num_joints * 2,
            num_joints * 3,
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
            p_dropout=None if args.task_dropout == 0.0 else args.task_dropout,
            # nodes_group=dataset.skeleton().joints_group() if args.non_local else None,
            nodes_group=(
                [[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]
                if args.non_local
                else None
            ),
        )
    elif args.task_net == "videopose":
        filter_widths = [1]
        for stage_id in range(4):  # args.stages
            filter_widths.append(1)  # filter_widths = [1, 1, 1, 1, 1]
        # filter_widths = [3, 3, 3, 3, 3]
        model_pos = TemporalModelOptimized1f(
            num_joints,
            2,
            num_joints,
            filter_widths=filter_widths,
            causal=False,
            dropout=0.25,
            channels=1024,
        )
    else:
        print(f"Invalid task-net: {args.task_net}")
        raise ValueError(f"Invalid task-net: {args.task_net}")

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        try:
            model_pos.load_state_dict(checkpoint["model"])
        except:
            model_pos = checkpoint

    print(
        f"==> Number of parameters: {sum(p.numel() for p in model_pos.parameters()):,}"
    )

    model_pos.to(device)
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model_pos.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

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
        if args.em_loss_type == "margin":
            criterion_loss = MarginBasedLoss(
                margin_ratio=args.margin_ratio, loss_type=args.em_loss
            )
        elif args.em_loss_type == "nce":
            criterion_loss = NCELoss(temperature=args.margin_ratio)

        if args.checkpoint_loss is not None:
            checkpoint = torch.load(args.checkpoint_loss, map_location=device)
            try:
                model_loss.load_state_dict(checkpoint["loss_net"])
            except:
                try:
                    model_loss.load_state_dict(checkpoint)
                except:
                    model_loss = checkpoint

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

    best_pelvis_aligned_mpjpe = (
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    )
    best_p_mpjpe = float("inf")
    best_mpjpe = float("inf")

    epoch = 1
    epochs = args.num_epoch if args.patience < 0 else -1
    early_stopping_counter = 0
    lr_task_net = args.lr
    
    set_seed(args.seed)
    # One iterator for the whole run. next(iter(loader)) rebuilds the sampler on
    # every call, and RandomSampler materialises a permutation of all 1.84M
    # training poses (~136 ms) to use 1024 of them -- the dominant term in SEAL's
    # 4.4x wall-clock over baseline.
    loss_iter = iter(train_loader_loss)

    while True:
        # for epoch in range(1, epochs + 1):
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_energy = AverageMeter()
        epoch_loss_energy_u = AverageMeter()
        epoch_loss_dir = AverageMeter()
        epoch_e_diff = AverageMeter()
        epoch_loss_loss_net = AverageMeter()
        e_diffs = []
        model_pos.train()
        for i, batch in enumerate(tqdm(train_loader)):
            targets_3d, inputs_2d = batch[0].to(device), batch[1].to(device)
            batch_size = targets_3d.size(0)
            targets_3d = center(targets_3d)
            if args.type == "baseline":
                outputs_3d = model_pos(inputs_2d)
                optimizer.zero_grad()
                loss_3d_pos = criterion(outputs_3d, targets_3d)
                epoch_loss_3d_pos.update(loss_3d_pos.item(), batch_size)

                loss_total = loss_3d_pos
                if criterion_dir is not None:
                    loss_dir = criterion_dir(outputs_3d, targets_3d)
                    epoch_loss_dir.update(loss_dir.item(), batch_size)
                    loss_total = loss_total + args.dir_weight * loss_dir

                loss_total.backward()
                optimizer.step()
            elif args.type == "dynamic":
                # update loss net
                model_loss.train()
                if args.task_eval_during_dynamic:
                    model_pos.eval()

                try:
                    loss_batch = next(loss_iter)
                except StopIteration:
                    loss_iter = iter(train_loader_loss)
                    loss_batch = next(loss_iter)
                targets_3d_l, inputs_2d_l = loss_batch[0].to(device), loss_batch[1].to(
                    device
                )
                targets_3d_l = center(targets_3d_l)

                with torch.no_grad():
                    input_combined = torch.cat([inputs_2d, inputs_2d_l], dim=0)
                    target_combined = torch.cat([targets_3d, targets_3d_l], dim=0)
                    outputs_cobined = model_pos(input_combined)
                energy_hat = model_loss(input_combined, outputs_cobined)
                energy_label = model_loss(input_combined, target_combined)
                e_diff = energy_hat[batch_size:] - energy_label[batch_size:]
                epoch_e_diff.update(e_diff.mean().item(), batch_size)

                e_diffs += (e_diff.detach().cpu().numpy()).tolist()

                optimizer_loss.zero_grad()
                loss_loss_net = criterion_loss(
                    outputs_cobined[batch_size:],
                    target_combined[batch_size:],
                    energy_hat[batch_size:],
                    energy_label[batch_size:],
                ) + criterion_loss(
                    outputs_cobined[:batch_size],
                    target_combined[:batch_size],
                    energy_hat[:batch_size],
                    energy_label[:batch_size],
                )
                loss_loss_net.backward()
                optimizer_loss.step()

                epoch_loss_loss_net.update(loss_loss_net.item(), batch_size)

                # update task net
                model_pos.train()
                if args.loss_eval_during_dynamic:
                    model_loss.eval()
                outputs_3d = model_pos(inputs_2d)
                energy_hat = model_loss(inputs_2d, outputs_3d)

                loss_3d_pos = criterion(outputs_3d, targets_3d)
                loss_energy = (
                    energy_hat.mean()
                    * args.energy_weight
                    * (epoch / epochs if args.anneal and epochs > 0 else 1)
                )

                epoch_loss_3d_pos.update(loss_3d_pos.item(), batch_size)
                epoch_loss_energy.update(loss_energy.item(), batch_size)

                loss_with_energy = args.mse_weight * loss_3d_pos + loss_energy

                optimizer.zero_grad()
                loss_with_energy.backward()
                optimizer.step()

                # check nan - outputs_3d
                if torch.isnan(outputs_3d).any():
                    print(f"Epoch: {epoch}, Nan in outputs_3d")
                    raise ValueError("Nan in outputs_3d")

            elif args.type == "static":
                model_loss.eval()
                outputs_3d = model_pos(inputs_2d)
                energy_hat = model_loss(inputs_2d, outputs_3d)
                energy_label = model_loss(inputs_2d, targets_3d)
                e_diff = energy_hat - energy_label
                epoch_e_diff.update(e_diff.mean().item(), batch_size)
                e_diffs += (e_diff.detach().cpu().numpy()).tolist()
                optimizer.zero_grad()
                loss_energy = energy_hat.mean() * args.energy_weight
                loss_3d_pos = criterion(outputs_3d, targets_3d)
                loss_with_energy = loss_3d_pos + loss_energy
                loss_with_energy.backward()
                # loss_energy.backward()
                epoch_loss_3d_pos.update(loss_3d_pos.item(), batch_size)
                epoch_loss_energy.update(loss_energy.item(), batch_size)
                nn.utils.clip_grad_norm_(model_loss.parameters(), max_norm=1)
                optimizer.step()
            else:
                print(f"Invalid type: {args.type}")
                raise ValueError(f"Invalid type: {args.type}")

        lr_task_net = lr_task_net * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_task_net

        if epoch % args.eval_interval == 0:
            if args.dataset == "3dhp":
                error_h3wb_p1, error_h3wb_p2, pelvis_aligned_mpjpe, pck, auc = evaluate(
                    valid_loader, model_pos, device, is_h3wb=is_h3wb, base_joint=14 if args.dataset == "3dhp" else 0
                )
            else:
                error_h3wb_p1, error_h3wb_p2, pelvis_aligned_mpjpe = evaluate(
                    valid_loader, model_pos, device, is_h3wb=is_h3wb, base_joint=14 if args.dataset == "3dhp" else 0
                )
            if (
                error_h3wb_p1 < best_mpjpe and (not is_h3wb or not args.eval_by_pa)
            ) or (
                is_h3wb
                and args.eval_by_pa
                and pelvis_aligned_mpjpe[0] < best_pelvis_aligned_mpjpe[0]
            ):
                best_mpjpe = error_h3wb_p1
                best_p_mpjpe = error_h3wb_p2
                if is_h3wb:
                    best_pelvis_aligned_mpjpe = pelvis_aligned_mpjpe

                if save_dir is not None:
                    for file in os.listdir(save_dir):
                        if "best_tasknet" in file:
                            os.remove(os.path.join(save_dir, file))
                    torch.save(
                        model_pos,
                        os.path.join(save_dir, f"best_tasknet_{epoch}.pth"),
                    )
                    print(f"Model saved at {save_dir}, epoch {epoch}")
                early_stopping_counter = 0

                if run is not None:
                    if is_h3wb:
                        if (
                            (args.task_net == "linear" and pelvis_aligned_mpjpe[0] < 64)
                            or (
                                args.task_net == "semgcn"
                                and pelvis_aligned_mpjpe[0] < 60
                            )
                            or (
                                args.task_net == "videopose"
                                and pelvis_aligned_mpjpe[0] < 59.5
                            )
                        ):
                            run.alert(
                                title=f"H3WB {args.task_net} {args.type}",
                                text=f"PA-MPJPE {pelvis_aligned_mpjpe[0]:.2f} (Epoch {epoch})",
                                level=wandb.AlertLevel.INFO,
                            )
                    if not is_h3wb and error_h3wb_p1 < 42:
                        run.alert(
                            title=f"H36M {args.type}",
                            text=f"MPJPE {error_h3wb_p1:.2f} (Epoch {epoch})",
                            level=wandb.AlertLevel.INFO,
                        )
            else:
                early_stopping_counter += 1

            print(
                f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss_3d_pos.avg:.6f}, Time taken: {time.time() - start_time:.2f}s, Early stopping: {early_stopping_counter}"
            )
            if args.type != "baseline":
                print(
                    f"  Energy loss: {epoch_loss_energy.avg:.3E}, E-diff: {epoch_e_diff.avg:.6f}, E-diff Ratio: {np.mean(e_diffs)/np.std(e_diffs):.3f}"
                )
            if criterion_dir is not None:
                print(
                    f"  Dir loss: {epoch_loss_dir.avg:.6f} (x{args.dir_weight} = "
                    f"{args.dir_weight * epoch_loss_dir.avg:.6f}), pos loss: {epoch_loss_3d_pos.avg:.6f}"
                )
            print(
                f"{args.dataset}: Protocol #1   (MPJPE) overall average: {error_h3wb_p1:.2f} (mm)"
            )
            print(
                f"{args.dataset}: Protocol #2 (P-MPJPE) overall average: {error_h3wb_p2:.2f} (mm)"
            )
            if args.dataset == "3dhp":
                print(f"pck: {pck:.2f}, auc: {auc:.2f}")
            if is_h3wb:
                print(tuple(map(lambda x: round(x, 2), pelvis_aligned_mpjpe)))

            if run is not None:
                log = {
                    "loss": epoch_loss_3d_pos.avg,
                    "MPJPE": error_h3wb_p1,
                    "P-MPJPE": error_h3wb_p2,
                    "Best-P-MPJPE": best_p_mpjpe,
                    "Best-MPJPE": best_mpjpe,
                    "early-stopping": early_stopping_counter,
                }
                if is_h3wb:
                    log.update(
                        {
                            "PA-MPJPE-wb": pelvis_aligned_mpjpe[0],
                            "PA-MPJPE-b": pelvis_aligned_mpjpe[1],
                            "PA-MPJPE-f": pelvis_aligned_mpjpe[2],
                            "PA-MPJPE-h": pelvis_aligned_mpjpe[3],
                            "PA-MPJPE-fn": pelvis_aligned_mpjpe[4],
                            "PA-MPJPE-hw": pelvis_aligned_mpjpe[5],
                            "Best-PA-MPJPE-wb": best_pelvis_aligned_mpjpe[0],
                            "Best-PA-MPJPE-b": best_pelvis_aligned_mpjpe[1],
                            "Best-PA-MPJPE-f": best_pelvis_aligned_mpjpe[2],
                            "Best-PA-MPJPE-h": best_pelvis_aligned_mpjpe[3],
                            "Best-PA-MPJPE-fn": best_pelvis_aligned_mpjpe[4],
                            "Best-PA-MPJPE-hw": best_pelvis_aligned_mpjpe[5],
                        }
                    )
                if args.type != "baseline":
                    try:
                        log.update(
                            {
                                "Energy": epoch_loss_energy.avg,
                                "E-diff": epoch_e_diff.avg,
                                "E-diff-R": np.mean(e_diffs) / np.std(e_diffs),
                                "loss_em": epoch_loss_loss_net.avg,
                            }
                        )
                    except:
                        pass
                run.log(log)
            start_time = time.time()
            print(f"")

        if args.patience < 0 and epoch == epochs:
            break
        if args.patience >= 1 and early_stopping_counter >= args.patience:
            print("Early stopping")
            break
        if (
            args.dataset == "h3wb"
            and early_stopping_counter >= 50
            and not args.eval_by_pa
        ):
            print("Early stopping")
            raise ValueError("Early stopping")
        epoch += 1

    if epochs % args.eval_interval != 0:
        error_h3wb_p1, error_h3wb_p2, pelvis_aligned_mpjpe = evaluate(
            valid_loader, model_pos, device, is_h3wb=is_h3wb
        )
        print(
            f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss_3d_pos.avg:.6f}, Time taken: {time.time() - start_time:.2f}s"
        )
        print(f"{args.dataset}: Protocol #1   (MPJPE) overall average: {error_h3wb_p1:.2f} (mm)")
        print(f"{args.dataset}: Protocol #2 (P-MPJPE) overall average: {error_h3wb_p2:.2f} (mm)")
        if is_h3wb:
            print(tuple(map(lambda x: round(x, 2), pelvis_aligned_mpjpe)))
        if (not args.eval_by_pa and error_h3wb_p1 < best_mpjpe) or (
            is_h3wb
            and args.eval_by_pa
            and pelvis_aligned_mpjpe[0] < best_pelvis_aligned_mpjpe[0]
        ):
            best_mpjpe = error_h3wb_p1
            best_p_mpjpe = error_h3wb_p2
            if is_h3wb:
                best_pelvis_aligned_mpjpe = pelvis_aligned_mpjpe
            if save_dir is not None:
                torch.save(
                    model_pos,
                    os.path.join(save_dir, f"{epoch}-epoch_task_net.pth"),
                )
                print(f"Model saved at {save_dir}, epoch {epoch}")
            print(f"Model saved at {save_dir}, epoch {epoch}")

    if run is not None:
        log = {
            "loss": epoch_loss_3d_pos.avg,
            "MPJPE": error_h3wb_p1,
            "P-MPJPE": error_h3wb_p2,
            "Best-P-MPJPE": best_p_mpjpe,
            "Best-MPJPE": best_mpjpe,
        }
        if is_h3wb:
            log.update(
                {
                    "PA-MPJPE-wb": pelvis_aligned_mpjpe[0],
                    "PA-MPJPE-b": pelvis_aligned_mpjpe[1],
                    "PA-MPJPE-f": pelvis_aligned_mpjpe[2],
                    "PA-MPJPE-h": pelvis_aligned_mpjpe[3],
                    "PA-MPJPE-fn": pelvis_aligned_mpjpe[4],
                    "PA-MPJPE-hw": pelvis_aligned_mpjpe[5],
                    "Best-PA-MPJPE-wb": best_pelvis_aligned_mpjpe[0],
                    "Best-PA-MPJPE-b": best_pelvis_aligned_mpjpe[1],
                    "Best-PA-MPJPE-f": best_pelvis_aligned_mpjpe[2],
                    "Best-PA-MPJPE-h": best_pelvis_aligned_mpjpe[3],
                    "Best-PA-MPJPE-fn": best_pelvis_aligned_mpjpe[4],
                    "Best-PA-MPJPE-hw": best_pelvis_aligned_mpjpe[5],
                }
            )
        run.log(log)

    print(f"Best MPJPE on {args.dataset.upper()}: {best_mpjpe:.2f} (mm)")
    print(f"Best P-MPJPE on {args.dataset.upper()}: {best_p_mpjpe:.2f} (mm)")
    if is_h3wb:
        print(
            f"Best pelvis-aligned-MPJPE on H3WB: {tuple(map(lambda x: round(x, 2), best_pelvis_aligned_mpjpe))} (mm)\n"
        )

    if save_dir is not None:
        torch.save(
            {
                "model": model_pos.state_dict(),
                "loss_net": (
                    model_loss.state_dict() if args.type != "baseline" else None
                ),
            },
            args.save_path,
        )
        if args.type != "baseline":
            torch.save(model_loss, os.path.join(save_dir, f"final_loss_net.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--loss_batch_size", type=int, default=64)
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
    parser.add_argument("--checkpoint_loss", type=str, default=None)
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
    parser.add_argument("--task_eval_during_dynamic", type=int, default=1)
    parser.add_argument("--loss_eval_during_dynamic", type=int, default=1)
    parser.add_argument("--eval_by_pa", type=int, default=0)
    parser.add_argument("--pos_loss", type=str, default="mse")
    parser.add_argument("--em_loss", type=str, default="mse")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--wandb_tag", type=str, default=None)
    parser.add_argument("--anneal", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--non_local", type=int, default=0)
    parser.add_argument("--em_loss_type", type=str, default="margin")
    parser.add_argument("--centering", type=str, default=None)
    parser.add_argument("--mse_weight", type=float, default=1)
    parser.add_argument("--lr_decay", type=float, default=1)
    parser.add_argument("--dir_weight", type=float, default=0,
                        help="weight of the length-weighted bone-direction loss "
                             "(0 disables it); same units as the MSE term")
    parser.add_argument("--half", type=int, default=0)

    args = parser.parse_args()
    if args.dataset == "h3wb":
        args.eval_by_pa = 1
    print(args)

    random_seed = args.seed
    set_seed(random_seed)

    main(args)
