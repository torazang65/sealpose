"""Pretrain the unconditional energy prior with multi-scale DSM on 3DHP.

Stands alone from task training: the energy never sees 2D inputs or the task
net, so one pretraining run serves every downstream --type dsm-prior sweep.

Two representations (models/energy_net.py):

  --features joint   root-centred joint coordinates in metres, no
                     standardisation. The sigma ladder is in metres and was
                     set from the task net's root-centred residual, measured
                     2026-09-22 on gcn-base-ht-clean: per-coordinate RMS
                     ~250 mm at epoch 1 (= the GT spread, the prediction is
                     random), 60-90 mm after 3-5 epochs, 22 mm converged
                     (S8), ~34 mm at test time. Default ladder 5-250 mm,
                     sigma0 30 mm.
  --features bone    standardised bone vectors; ladder in standardised units,
                     defaults reproduce the 2026-08-26 report.

Per-sigma validation diagnostics, printed every epoch alongside the loss:
cos(sigma0^2 grad E, x~ - x) and the one-step denoising ratio. Both come out
of the gradient the loss already computes, so they cost nothing extra. The
memorisation check the sigma floor depends on is the loss at the smallest
rung: if it climbs from epoch 1 while the others fall, the floor is below
the skeleton gap and must go up.

Split policy (do not "fix" this to match the task net): the task net has no
validation split -- its per-epoch numbers come from the TS1-TS6 *test* set.
Early-stopping the prior there would select the prior on test data and poison
every downstream comparison, so stopping is decided on held-out *training*
subjects (default S8) and TS never enters this script. The training subjects
and the 2D blow-up frame drop match the task net exactly, so both models see
the same training distribution. For the retrain-on-full variant, pass
--val_subjects "" with a fixed --num_epoch (no early stopping).

Usage:
    python train_energy_dsm.py --save_path checkpoints/energy-joint/best_energy.pth
    sbatch --job-name=energy-joint slurm/pyjob.sbatch train_energy_dsm.py \
        --save_path checkpoints/energy-joint/best_energy.pth
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
                               RootCentered, Standardizer,
                               save_energy_checkpoint)
from utils.data_utils import (create_2d_data, drop_extreme_2d, fetch_h36m,
                              read_3d_data_3dhp)
from utils.device import get_device
from utils.seed import set_seed
from utils.utils import AverageMeter

SIGMA_DEFAULTS = {          # (sigma_min, sigma_max, sigma0)
    "joint": (0.005, 0.25, 0.03),   # metres
    "bone": (0.05, 2.0, 0.1),       # standardised units
}


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


def coordinate_statistics(poses_3d, transform, device, chunk=65536):
    """Per-coordinate mean/std of the flattened representation over the split."""
    all_poses = np.concatenate(poses_3d).astype(np.float32)
    total = all_poses.shape[0]
    s = s2 = None
    for start in range(0, total, chunk):
        y = torch.from_numpy(all_poses[start:start + chunk]).to(device)
        z = transform(y).flatten(1).double()
        if s is None:
            s = torch.zeros(z.shape[1], dtype=torch.float64, device=device)
            s2 = torch.zeros_like(s)
        s += z.sum(dim=0)
        s2 += z.pow(2).sum(dim=0)
    mean = s / total
    var = (s2 / total - mean.pow(2)).clamp_min(0)
    return mean.float().cpu(), var.sqrt().float().cpu()


class PerSigma:
    """Accumulates per-sample diagnostics into per-sigma means."""

    def __init__(self, num_sigmas, keys):
        self.sum = {k: torch.zeros(num_sigmas) for k in keys}
        self.count = torch.zeros(num_sigmas)

    def update(self, idx, diagnostics):
        idx = idx.cpu()
        for k, v in diagnostics.items():
            self.sum[k].index_add_(0, idx, v.cpu())
        self.count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    def mean(self):
        return {k: v / self.count.clamp_min(1) for k, v in self.sum.items()}


def run_epoch(loader, corrupt_space, net, dsm, optimizer=None, generator=None,
              grad_clip=0.0, limit_batches=0, device="cpu"):
    """One pass; optimizer=None means evaluation (fresh generator => same noise).

    corrupt_space maps a pose batch to the coordinates the noise is added to,
    which for both representations is also the net's input.
    """
    meter = AverageMeter()
    per_sigma = PerSigma(dsm.sigmas.numel(), ("loss", "cos", "one_step"))
    training = optimizer is not None
    net.train(training)
    for i, y in enumerate(loader):
        y = y.to(device)
        x = corrupt_space(y)
        loss, idx, diagnostics = dsm(net, x, generator=generator,
                                     create_graph=training)
        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            optimizer.step()
        meter.update(loss.item(), y.shape[0])
        per_sigma.update(idx, diagnostics)
        if limit_batches and i + 1 >= limit_batches:
            break
    return meter.avg, per_sigma.mean()


def format_per_sigma(sigmas, values, fmt=".3f"):
    return " ".join(f"{s:.3g}:{v:{fmt}}" for s, v in zip(sigmas.tolist(),
                                                         values.tolist()))


def print_per_sigma(label, sigmas, stats):
    print(f"  {label} loss:     {format_per_sigma(sigmas, stats['loss'])}")
    print(f"  {label} cos:      {format_per_sigma(sigmas, stats['cos'])}")
    print(f"  {label} one-step: {format_per_sigma(sigmas, stats['one_step'])}")


def main(args):
    device = get_device()
    print(f"==> Device: {device}")

    print("==> Preparing data...")
    h36m_dataset = MpiInf3dhpDataset(args.data_path)
    # The skeleton needs slot 0 to be a real joint (head_top), not the
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
    sigmas = torch.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.num_sigmas
    )
    mean = std = None
    if args.features == "joint":
        transform = RootCentered(parents).to(device)
        corrupt_space = transform
        in_dim = transform.dim
        print(f"==> Joints: {len(parents)}, root: joint {transform.root}, "
              f"root-centred coordinates (m): {in_dim}")
        _, spread = coordinate_statistics(poses_train, transform, device)
        print(f"==> Coordinate std (m): min {spread.min():.4f}, "
              f"median {spread.median():.4f}, max {spread.max():.4f}")
        print(f"==> Sigma ladder (mm): "
              f"{[round(s * 1000, 1) for s in sigmas.tolist()]}, "
              f"sigma0 {args.sigma0 * 1000:.1f} mm")
    else:
        transform = BoneTransform(parents).to(device)
        in_dim = transform.num_bones * 3
        print(f"==> Joints: {len(parents)}, bones: {transform.num_bones}, "
              f"root: joint {transform.root}")
        mean, std = coordinate_statistics(poses_train, transform, device)
        standardizer = Standardizer(mean, std).to(device)
        corrupt_space = lambda y: standardizer(transform(y).flatten(1))
        print(f"==> Bone-vector std (m): min {std.min():.4f}, "
              f"median {std.median():.4f}, max {std.max():.4f}")
        print(f"==> Sigma ladder (standardized): "
              f"{[round(s, 4) for s in sigmas.tolist()]}")
        print(f"==> Sigma ladder (~mm at median std): "
              f"{[round(s * std.median().item() * 1000, 1) for s in sigmas.tolist()]}, "
              f"sigma0 {args.sigma0}")

    dsm = MultiScaleDSM(sigmas, args.sigma0).to(device)
    net = EnergyNet(in_dim, hidden=args.hidden, depth=args.depth).to(device)
    print(f"==> Number of parameters (energy net): "
          f"{sum(p.numel() for p in net.parameters()):,}")
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    def save(epoch, val_dsm):
        save_energy_checkpoint(
            args.save_path, net, parents, sigmas, args.sigma0,
            features=args.features, mean=mean, std=std,
            train_args=vars(args), epoch=epoch, val_dsm=val_dsm,
        )

    best_val = float("inf")
    best_epoch = 0
    early_stopping_counter = 0
    start_time = time.time()
    for epoch in range(1, args.num_epoch + 1):
        train_loss, train_stats = run_epoch(
            train_loader, corrupt_space, net, dsm,
            optimizer=optimizer, grad_clip=args.grad_clip,
            limit_batches=args.limit_train_batches, device=device,
        )

        if val_loader is not None:
            # Fresh generator with a fixed seed: identical noise every eval,
            # so patience reacts to the model, not to the noise draw.
            generator = torch.Generator().manual_seed(args.seed + 10000)
            val_loss, val_stats = run_epoch(
                val_loader, corrupt_space, net, dsm,
                optimizer=None, generator=generator,
                limit_batches=args.limit_val_batches, device=device,
            )
        else:
            val_loss, val_stats = train_loss, train_stats

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            early_stopping_counter = 0
            save(epoch, val_loss)
        else:
            early_stopping_counter += 1

        print(f"Epoch [{epoch}/{args.num_epoch}], DSM: {train_loss:.6E}, "
              f"val DSM: {val_loss:.6E}, Time taken: {time.time() - start_time:.2f}s, "
              f"Early stopping: {early_stopping_counter}"
              + (" *saved*" if improved else ""))
        print(f"  train loss:     {format_per_sigma(sigmas, train_stats['loss'])}")
        if val_loader is not None:
            print_per_sigma("val", sigmas, val_stats)
        start_time = time.time()

        if args.patience >= 1 and early_stopping_counter >= args.patience:
            print("Early stopping")
            break

    if val_loader is None:
        # retrain-on-full mode: no selection signal, the final epoch is the model
        save(args.num_epoch, None)
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
    parser.add_argument("--features", type=str, default="joint",
                        choices=["joint", "bone"],
                        help="energy input: root-centred joints in metres, or "
                             "standardised bone vectors (the 2026-08-26 "
                             "report's representation)")
    parser.add_argument("--sigma_min", type=float, default=None,
                        help="smallest noise scale; metres for joint, "
                             "standardised units for bone. Defaults: "
                             "0.005 / 0.05")
    parser.add_argument("--sigma_max", type=float, default=None,
                        help="largest noise scale; defaults 0.25 / 2.0")
    parser.add_argument("--num_sigmas", type=int, default=12)
    parser.add_argument("--sigma0", type=float, default=None,
                        help="MDSM anchor scale in the gradient term; it only "
                             "rescales E, and gate 4 recalibrates "
                             "energy_weight against it. Defaults: 0.03 / 0.1")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--limit_train_batches", type=int, default=0,
                        help="smoke testing only; 0 = full epoch")
    parser.add_argument("--limit_val_batches", type=int, default=0)

    args = parser.parse_args()
    # Sigma lives in different units per representation, so the ladder cannot
    # have one default.
    lo, hi, s0 = SIGMA_DEFAULTS[args.features]
    if args.sigma_min is None:
        args.sigma_min = lo
    if args.sigma_max is None:
        args.sigma_max = hi
    if args.sigma0 is None:
        args.sigma0 = s0
    print(args)
    set_seed(args.seed)
    main(args)
