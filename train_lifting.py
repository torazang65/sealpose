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
from models.energy_net import EnergyPrior
from models.linear_model import LinearModel
from models.loss_net import LinearLossNet, MarginBasedLoss, NCELoss
from models.sem_gcn import SemGCN, adj_mx_from_skeleton
from models.video_pose import TemporalModelOptimized1f
from utils.camera import normalize_screen_coordinates
from utils.data_utils import (create_2d_data, drop_extreme_2d, fetch_h36m,
                             read_3d_data, read_3d_data_3dhp)
from utils.device import get_device
from utils.h36m_dataset import Human36mDataset

from torch.utils.data import DataLoader

from utils.loss import (BoneDirectionLoss, BoneDirectionPerturber,
                        GaussianPerturber, mpjpe)
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
    dataset = read_3d_data_3dhp(h36m_dataset, restore_head_top=args.restore_head_top)
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
    if args.max_2d_abs > 0:
        (
            poses_train,
            poses_train_2d,
            actions_train,
            cams_train,
            dropped,
        ) = drop_extreme_2d(
            poses_train, poses_train_2d, actions_train, cams_train, args.max_2d_abs
        )
        print(
            f"==> Dropped {dropped} training frames with max|2d| > {args.max_2d_abs} "
            f"(invalid projections; evaluation split untouched)"
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

    def prior_weight(epoch):
        """energy_weight for this epoch under the dsm-prior schedule.

        warm-up (0) -> fixed energy_weight -> linear ramp to 0 over
        [anneal_start, anneal_end] -> 0. With no anneal the weight stays fixed
        after warm-up. The two runs the schedule is for differ only in the
        ramp: if the gain survives the ramp the prior changed which basin the
        task net settled in; if it vanishes, the prior was a regulariser that
        held the net somewhere it does not stay on its own.
        """
        if epoch <= args.prior_warmup:
            return 0.0
        if args.prior_anneal_start <= 0:
            return args.energy_weight
        if epoch >= args.prior_anneal_end:
            return 0.0
        if epoch < args.prior_anneal_start:
            return args.energy_weight
        span = max(args.prior_anneal_end - args.prior_anneal_start, 1)
        return args.energy_weight * (args.prior_anneal_end - epoch) / span

    print("==> Data loaded...")
    device = get_device()
    if args.pos_loss == "mse":
        criterion = nn.MSELoss(reduction="mean").to(device)
    elif args.pos_loss == "mpjpe":
        criterion = mpjpe
    num_joints = dataset.skeleton().num_joints()
    print(f"==> Number of joints: {num_joints}")

    # 3DHP stores the camera-frame trajectory in slot 0 instead of head_top, so
    # that slot holds 1.2-6.3 m of camera distance rather than a body joint and
    # every skeleton-aware term has to drop it. --restore_head_top puts the real
    # joint back, at which point there is nothing to exclude.
    trajectory_joint = []
    if args.dataset == "3dhp" and not args.restore_head_top:
        trajectory_joint = [0]
    if args.dataset == "3dhp":
        print(f"==> Joint 0: {'head_top (restored)' if args.restore_head_top else 'camera trajectory (excluded from structural terms)'}")

    # Length-weighted bone-direction term (report/check_bodyness.py, section 7).
    criterion_dir = None
    if args.dir_weight > 0:
        criterion_dir = BoneDirectionLoss(
            dataset.skeleton().parents(),
            exclude=trajectory_joint,
        ).to(device)
        print(f"==> Bone-direction loss on {len(criterion_dir.child)} bones, "
              f"weight {args.dir_weight}")

    perturber = None
    criterion_loss_neg = None
    neg_exclude = list(trajectory_joint)
    if args.neg_type != "none" and args.type != "dynamic":
        raise ValueError("--neg_type only applies to --type dynamic")

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

    if args.type not in ("baseline", "dsm-prior"):
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

        # Synthetic negatives for the energy net. Until now its only negative
        # was the task net's own prediction, whose error shrinks as training
        # proceeds, so the margin collapses and the energy net ends up
        # regressing error magnitude rather than learning structure
        # (report/2026-08-18_bone_direction_loss.md, section 1). These add a
        # supply at a difficulty we control, along the direction axis that holds
        # 2.4x the headroom of length. They supplement the prediction negatives
        # rather than replacing them: predictions carry a -18 mm bone-length
        # contraction that synthetic poses never show, and dropping them would
        # leave the energy net off-distribution exactly where the task net lives.
        if args.neg_type != "none":
            joint_mask = torch.ones(num_joints)
            for joint in neg_exclude:
                joint_mask[joint] = 0
            if args.em_loss_type == "margin":
                criterion_loss_neg = MarginBasedLoss(
                    margin_ratio=args.margin_ratio,
                    loss_type=args.em_loss,
                    per_sample=True,
                    joint_mask=joint_mask,
                )
            else:
                criterion_loss_neg = NCELoss(temperature=args.margin_ratio)
            if args.neg_type == "angle":
                perturber = BoneDirectionPerturber(
                    dataset.skeleton().parents(),
                    exclude=neg_exclude,
                    theta_min=args.neg_theta_min,
                    theta_max=args.neg_theta_max,
                ).to(device)
                detail = f"theta U({args.neg_theta_min}, {args.neg_theta_max}) deg"
            elif args.neg_type == "gauss":
                parents = list(dataset.skeleton().parents())
                perturber = GaussianPerturber(
                    root=parents.index(-1),
                    exclude=neg_exclude,
                    sigma=args.neg_sigma,
                ).to(device)
                detail = f"sigma {args.neg_sigma}"
            else:
                print(f"Invalid neg_type: {args.neg_type}")
                raise ValueError(f"Invalid neg_type: {args.neg_type}")
            print(f"==> Synthetic negatives: {args.neg_type}, {detail}, "
                  f"weight {args.neg_weight}, joints {neg_exclude} pinned to GT")

        if args.checkpoint_loss is not None:
            checkpoint = torch.load(args.checkpoint_loss, map_location=device)
            try:
                model_loss.load_state_dict(checkpoint["loss_net"])
            except:
                try:
                    model_loss.load_state_dict(checkpoint)
                except:
                    model_loss = checkpoint

    energy_prior = None
    if args.type == "dsm-prior":
        if args.energy_checkpoint is None:
            raise ValueError("--type dsm-prior requires --energy_checkpoint "
                             "(pretrain one with train_energy_dsm.py)")
        energy_prior = EnergyPrior.load(args.energy_checkpoint, map_location=device)
        skeleton_parents = [int(p) for p in dataset.skeleton().parents()]
        if energy_prior.parents != skeleton_parents:
            raise ValueError(
                f"energy checkpoint skeleton {energy_prior.parents} does not "
                f"match dataset skeleton {skeleton_parents}; was it pretrained "
                f"with the same dataset and --restore_head_top?"
            )
        energy_prior.to(device)
        energy_prior.freeze()
        # Same line format as the loss-net so report/parse_logs.py keeps the
        # params_loss column filled for these runs.
        print(
            f"==> Number of parameters (loss-net): {sum(p.numel() for p in energy_prior.parameters()):,}"
        )
        print(f"==> DSM energy prior: {args.energy_checkpoint} (frozen), "
              f"features {energy_prior.features}, "
              f"pretrain epoch {energy_prior.meta.get('epoch')}, "
              f"val DSM {energy_prior.meta.get('val_dsm')}")

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
    #
    # Only for the runs that consume it. iter() draws a seed for the sampler's
    # generator off the global RNG, so building it unconditionally would shift
    # the stream for baseline runs, which never touch this loader, and they
    # would stop reproducing the numbers they were tuned against.
    # dsm-prior never consumes it either, and skipping keeps that run's RNG
    # stream identical to baseline's.
    loss_iter = iter(train_loader_loss) if args.type in ("dynamic", "static") else None

    while True:
        # for epoch in range(1, epochs + 1):
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_energy = AverageMeter()
        epoch_loss_energy_u = AverageMeter()
        epoch_loss_dir = AverageMeter()
        epoch_loss_neg = AverageMeter()
        epoch_delta_neg = AverageMeter()
        epoch_e_diff_neg = AverageMeter()
        epoch_e_diff = AverageMeter()
        epoch_loss_loss_net = AverageMeter()
        e_diffs = []
        energy_weight = prior_weight(epoch) if args.type == "dsm-prior" else args.energy_weight
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
                if perturber is not None:
                    with torch.no_grad():
                        y_neg = perturber(target_combined)
                    energy_neg = model_loss(input_combined, y_neg)
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
                if perturber is not None:
                    loss_neg = criterion_loss_neg(
                        y_neg, target_combined, energy_neg, energy_label
                    )
                    loss_loss_net = loss_loss_net + args.neg_weight * loss_neg
                    epoch_loss_neg.update(loss_neg.item(), batch_size)
                    epoch_e_diff_neg.update(
                        (energy_neg - energy_label).mean().item(), batch_size
                    )
                    if hasattr(criterion_loss_neg, "delta"):
                        epoch_delta_neg.update(
                            criterion_loss_neg.delta(y_neg, target_combined).mean().item(),
                            batch_size,
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

            elif args.type == "dsm-prior":
                # Frozen pretrained prior: no energy-net update, no loss
                # loader, no negatives -- the only extra term is E*(T(y_hat)),
                # whose input gradient B^T grad_z E is the DSM-trained field.
                outputs_3d = model_pos(inputs_2d)
                loss_3d_pos = criterion(outputs_3d, targets_3d)
                energy_hat = energy_prior(outputs_3d)
                with torch.no_grad():
                    energy_label = energy_prior(targets_3d)
                e_diff = energy_hat.detach() - energy_label
                epoch_e_diff.update(e_diff.mean().item(), batch_size)
                e_diffs += e_diff.cpu().numpy().ravel().tolist()

                loss_energy = energy_hat.mean() * energy_weight
                epoch_loss_3d_pos.update(loss_3d_pos.item(), batch_size)
                epoch_loss_energy.update(loss_energy.item(), batch_size)

                # At weight 0 (warm-up, post-anneal) skip the prior's backward
                # entirely so the step is exactly the baseline's, not
                # baseline plus a zero-scaled graph.
                loss_total = args.mse_weight * loss_3d_pos
                if energy_weight > 0:
                    loss_total = loss_total + loss_energy
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

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
                    + (f", energy_weight: {energy_weight:.3E}" if args.type == "dsm-prior" else "")
                )
            if perturber is not None:
                print(
                    f"  Neg ({args.neg_type}): hinge {epoch_loss_neg.avg:.6f} "
                    f"(x{args.neg_weight} = {args.neg_weight * epoch_loss_neg.avg:.6f}), "
                    f"delta {epoch_delta_neg.avg:.6f}, "
                    f"E-diff {epoch_e_diff_neg.avg:.6f}"
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
                                "energy_weight": energy_weight,
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
                    model_loss.state_dict()
                    if args.type not in ("baseline", "dsm-prior")
                    else None
                ),
            },
            args.save_path,
        )
        if args.type not in ("baseline", "dsm-prior"):
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
    parser.add_argument("--restore_head_top", action="store_true",
                        help="3dhp only: slot 0 stores the camera-frame "
                             "trajectory instead of head_top "
                             "(data/prepare_data_mpi_inf_3dhp.py:489). Zero it "
                             "at load time to get the real 17-joint body pose. "
                             "Changes the meaning of MPJPE/PCK/AUC, so results "
                             "are not comparable across this flag")
    parser.add_argument("--type", type=str, default="baseline")
    parser.add_argument("--no_logging", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_loss", type=str, default=None)
    parser.add_argument("--energy_checkpoint", type=str, default=None,
                        help="pretrained DSM energy prior for --type dsm-prior "
                             "(train_energy_dsm.py output); loaded frozen")
    parser.add_argument("--prior_warmup", type=int, default=0,
                        help="--type dsm-prior: epochs of pure supervised "
                             "training before the prior term switches on; the "
                             "early residual (~250 mm per coordinate at epoch "
                             "1) is a random pose, not a noised one")
    parser.add_argument("--prior_anneal_start", type=int, default=0,
                        help="--type dsm-prior: first epoch of the linear "
                             "ramp of energy_weight to zero; 0 keeps it fixed")
    parser.add_argument("--prior_anneal_end", type=int, default=0,
                        help="--type dsm-prior: epoch at which energy_weight "
                             "reaches zero; pure supervised from then on")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="h3wb")
    parser.add_argument("--task_net", type=str, default="linear")
    parser.add_argument("--task_dropout", type=float, default=0.5)
    parser.add_argument("--task_num_stage", type=int, default=2)
    parser.add_argument("--task_linear_size", type=int, default=1024)
    parser.add_argument("--task_batch_norm", type=int, default=1)
    parser.add_argument("--keypoints", type=str, default="gt")
    parser.add_argument("--max_2d_abs", type=float, default=10.0,
                        help="3dhp only: drop TRAINING frames whose 2D "
                             "annotation exceeds this magnitude. 3DHP's 2D "
                             "is the mocap 3D projected per camera, so a "
                             "diverging trajectory writes the perspective "
                             "divide to file unchecked (308 frames, up to "
                             "23364, on a screen normalised to [-1, 1]). "
                             "They poison BatchNorm running stats and spike "
                             "valid MPJPE. Set <= 0 to keep them and "
                             "reproduce runs made before this flag")
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
    parser.add_argument("--neg_type", type=str, default="none",
                        help="synthetic negatives for the SEAL energy net: none | "
                             "angle (bone-direction perturbation) | gauss "
                             "(Cartesian control, difficulty-matched to angle)")
    parser.add_argument("--neg_weight", type=float, default=1.0,
                        help="weight of the synthetic-negative hinge term")
    parser.add_argument("--neg_theta_min", type=float, default=3.0,
                        help="min bone rotation in degrees for --neg_type angle")
    parser.add_argument("--neg_theta_max", type=float, default=15.0,
                        help="max bone rotation in degrees for --neg_type angle")
    parser.add_argument("--neg_sigma", type=float, default=0.01832,
                        help="per-coordinate noise std in metres for --neg_type "
                             "gauss; calibrated so mean delta matches angle")
    parser.add_argument("--half", type=int, default=0)

    args = parser.parse_args()
    if args.dataset == "h3wb":
        args.eval_by_pa = 1
    print(args)

    random_seed = args.seed
    set_seed(random_seed)

    main(args)
