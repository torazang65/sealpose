"""Pretrain the unconditional energy prior E(z) with multi-scale DSM on 3DHP.

Stands alone from task training: the energy never sees 2D inputs or the task
net, so one pretraining run serves every downstream --type dsm-prior sweep.

Split policy (do not "fix" this to match the task net): the task net has no
validation split -- its per-epoch numbers come from the TS1-TS6 *test* set.
Early-stopping the prior there would select the prior on test data and poison
every downstream comparison, so stopping is decided on held-out *training*
subjects (default S8) and TS never enters this script. The training subjects
and the 2D blow-up frame drop match the task net exactly, so both models see
the same training distribution. For the retrain-on-full variant, pass
--val_subjects "" with a fixed --num_epoch (no early stopping).

Usage:
    python train_energy_dsm.py --save_path checkpoints/energy-dsm/best_energy.pth
    sbatch --job-name=energy-dsm slurm/pyjob.sbatch train_energy_dsm.py \
        --save_path checkpoints/energy-dsm/best_energy.pth
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.prepare_data_mpi_inf_3dhp import MpiInf3dhpDataset
from data_loader import PoseTarget3D
from models.energy_net import (BoneTransform, EnergyNet, MultiScaleDSM,
                               Standardizer, save_energy_checkpoint)
from utils.data_utils import (create_2d_data, drop_extreme_2d, fetch_h36m,
                              read_3d_data_3dhp)
from utils.device import get_device
from utils.seed import set_seed
from utils.utils import AverageMeter


def load_split_poses(args, dataset, keypoints, subjects):
    """3D poses for a subject list, with the same frame drop as task training.

    The 308 blow-up frames are invalid 2D *annotations* -- the 3D pose itself
    survives -- but the task net never trains on them, and the prior should
    match the task net's training distribution rather than quietly differ by
    308 frames.
    """
    poses_3d, poses_2d, actions, cams = fetch_h36m(
        subjects, dataset, keypoints, action_filter=None, stride=1
    )
    if args.max_2d_abs > 0:
        poses_3d, poses_2d, actions, cams, dropped = drop_extreme_2d(
            poses_3d, poses_2d, actions, cams, args.max_2d_abs
        )
        print(f"==> Dropped {dropped} frames with max|2d| > {args.max_2d_abs} "
              f"from {subjects}")
    return poses_3d


def bone_statistics(poses_3d, bones, device, chunk=65536):
    """Per-coordinate mean/std of flattened bone vectors over the full split."""
    all_poses = np.concatenate(poses_3d).astype(np.float32)
    total = all_poses.shape[0]
    dim = bones.num_bones * 3
    s = torch.zeros(dim, dtype=torch.float64, device=device)
    s2 = torch.zeros(dim, dtype=torch.float64, device=device)
    for start in range(0, total, chunk):
        y = torch.from_numpy(all_poses[start:start + chunk]).to(device)
        z = bones(y).flatten(1).double()
        s += z.sum(dim=0)
        s2 += z.pow(2).sum(dim=0)
    mean = s / total
    var = (s2 / total - mean.pow(2)).clamp_min(0)
    return mean.float().cpu(), var.sqrt().float().cpu()


def run_epoch(loader, bones, standardizer, net, dsm, optimizer=None,
              generator=None, grad_clip=0.0, limit_batches=0, device="cpu"):
    """One pass; optimizer=None means evaluation (fresh generator => same noise)."""
    meter = AverageMeter()
    num_sigmas = dsm.sigmas.numel()
    per_sigma_sum = torch.zeros(num_sigmas)
    per_sigma_count = torch.zeros(num_sigmas)
    training = optimizer is not None
    net.train(training)
    for i, y in enumerate(loader):
        y = y.to(device)
        z = standardizer(bones(y).flatten(1))
        loss, idx, per_sample = dsm(net, z, generator=generator,
                                    create_graph=training)
        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            optimizer.step()
        meter.update(loss.item(), y.shape[0])
        idx = idx.cpu()
        per_sigma_sum.index_add_(0, idx, per_sample.cpu())
        per_sigma_count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        if limit_batches and i + 1 >= limit_batches:
            break
    per_sigma = per_sigma_sum / per_sigma_count.clamp_min(1)
    return meter.avg, per_sigma


def format_per_sigma(sigmas, per_sigma):
    return " ".join(f"{s:.3g}:{v:.3f}" for s, v in zip(sigmas.tolist(),
                                                       per_sigma.tolist()))


def main(args):
    device = get_device()
    print(f"==> Device: {device}")

    print("==> Preparing data...")
    h36m_dataset = MpiInf3dhpDataset(args.data_path)
    # The bone tree needs slot 0 to be a real joint (head_top), not the
    # camera-frame trajectory, so restore_head_top is not optional here.
    dataset = read_3d_data_3dhp(h36m_dataset, restore_head_top=True)
    print("==> Loading 2D detections...")
    keypoints = create_2d_data(
        f"data/data_2d_mpi_inf_3dhp_{args.keypoints}.npz", dataset
    )

    train_subjects = [s for s in args.subjects.split(",") if s]
    val_subjects = [s for s in args.val_subjects.split(",") if s]
    overlap = set(train_subjects) & set(val_subjects)
    if overlap:
        raise ValueError(f"subjects appear in both splits: {sorted(overlap)}")
    print(f"==> Train subjects: {train_subjects}, val subjects: {val_subjects or 'none'}")

    poses_train = load_split_poses(args, dataset, keypoints, train_subjects)
    train_loader = DataLoader(
        PoseTarget3D(poses_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = None
    if val_subjects:
        poses_val = load_split_poses(args, dataset, keypoints, val_subjects)
        val_loader = DataLoader(
            PoseTarget3D(poses_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    parents = list(dataset.skeleton().parents())
    bones = BoneTransform(parents).to(device)
    print(f"==> Joints: {len(parents)}, bones: {bones.num_bones}, "
          f"root: joint {parents.index(-1)}")

    mean, std = bone_statistics(poses_train, bones, device)
    standardizer = Standardizer(mean, std).to(device)
    print(f"==> Bone-vector std (m): min {std.min():.4f}, "
          f"median {std.median():.4f}, max {std.max():.4f}")

    sigmas = torch.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.num_sigmas
    )
    dsm = MultiScaleDSM(sigmas, args.sigma0).to(device)
    print(f"==> Sigma ladder (standardized): "
          f"{[round(s, 4) for s in sigmas.tolist()]}")
    print(f"==> Sigma ladder (~mm at median std): "
          f"{[round(s * std.median().item() * 1000, 1) for s in sigmas.tolist()]}, "
          f"sigma0 {args.sigma0}")

    net = EnergyNet(bones.num_bones * 3, hidden=args.hidden, depth=args.depth).to(device)
    print(f"==> Number of parameters (energy net): "
          f"{sum(p.numel() for p in net.parameters()):,}")
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    early_stopping_counter = 0
    start_time = time.time()
    for epoch in range(1, args.num_epoch + 1):
        train_loss, train_per_sigma = run_epoch(
            train_loader, bones, standardizer, net, dsm,
            optimizer=optimizer, grad_clip=args.grad_clip,
            limit_batches=args.limit_train_batches, device=device,
        )

        if val_loader is not None:
            # Fresh generator with a fixed seed: identical noise every eval,
            # so patience reacts to the model, not to the noise draw.
            generator = torch.Generator().manual_seed(args.seed + 10000)
            val_loss, val_per_sigma = run_epoch(
                val_loader, bones, standardizer, net, dsm,
                optimizer=None, generator=generator,
                limit_batches=args.limit_val_batches, device=device,
            )
        else:
            val_loss, val_per_sigma = train_loss, train_per_sigma

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            early_stopping_counter = 0
            save_energy_checkpoint(
                args.save_path, net, parents, mean, std, sigmas, args.sigma0,
                train_args=vars(args), epoch=epoch, val_dsm=val_loss,
            )
        else:
            early_stopping_counter += 1

        print(f"Epoch [{epoch}/{args.num_epoch}], DSM: {train_loss:.6E}, "
              f"val DSM: {val_loss:.6E}, Time taken: {time.time() - start_time:.2f}s, "
              f"Early stopping: {early_stopping_counter}"
              + (" *saved*" if improved else ""))
        print(f"  per-sigma train: {format_per_sigma(sigmas, train_per_sigma)}")
        if val_loader is not None:
            print(f"  per-sigma val:   {format_per_sigma(sigmas, val_per_sigma)}")
        start_time = time.time()

        if args.patience >= 1 and early_stopping_counter >= args.patience:
            print("Early stopping")
            break

    if val_loader is None:
        # retrain-on-full mode: no selection signal, the final epoch is the model
        save_energy_checkpoint(
            args.save_path, net, parents, mean, std, sigmas, args.sigma0,
            train_args=vars(args), epoch=args.num_epoch, val_dsm=None,
        )
        print(f"Saved final epoch to {args.save_path} (no validation split)")
    else:
        print(f"Best val DSM: {best_val:.6E} (epoch {best_epoch}), "
              f"saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str,
                        default="data/data_3d_mpi_inf_3dhp.npz")
    parser.add_argument("--keypoints", type=str, default="gt")
    parser.add_argument("--subjects", type=str, default="S1,S2,S3,S4,S5,S6,S7",
                        help="training subjects (comma separated)")
    parser.add_argument("--val_subjects", type=str, default="S8",
                        help="held-out TRAINING subjects for early stopping; "
                             "empty string trains on --subjects for exactly "
                             "--num_epoch epochs and saves the final model "
                             "(retrain-on-full mode). Never TS1-TS6: the test "
                             "set must not select the prior")
    parser.add_argument("--max_2d_abs", type=float, default=10.0,
                        help="same blow-up frame drop as train_lifting.py, so "
                             "the prior trains on the task net's distribution")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="max grad norm for the energy net; 0 disables")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10,
                        help="epochs without val improvement before stopping; "
                             "<1 disables early stopping")
    parser.add_argument("--sigma_min", type=float, default=0.01,
                        help="smallest noise scale, standardized units")
    parser.add_argument("--sigma_max", type=float, default=2.0,
                        help="largest noise scale, standardized units")
    parser.add_argument("--num_sigmas", type=int, default=12)
    parser.add_argument("--sigma0", type=float, default=0.1,
                        help="MDSM anchor scale in the gradient term")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--limit_train_batches", type=int, default=0,
                        help="smoke testing only; 0 = full epoch")
    parser.add_argument("--limit_val_batches", type=int, default=0)

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    main(args)
