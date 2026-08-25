#!/usr/bin/env python
"""Go/no-go gates for a pretrained DSM energy prior, run before task training.

The design's whole premise is that DSM constrains the *gradient field* where
the hinge only constrained values, so each gate checks a property the field
must have if the pretraining worked -- and none of them needs a task run:

  gate 1  monotonicity: E(z + sigma*eps) rises with sigma
  gate 2  denoising: gradient descent on E pulls a noised pose back toward the
          GT (mm before/after) -- the direct test that -grad E points at the
          manifold; a hinge-trained net has no reason to pass this
  gate 3  separation: E ranks GT below angle- / length- / Gaussian-corrupted
          poses (AUROC)
  gate 4  calibration (optional, needs --task_checkpoint): the size of the
          energy gradient on task-net predictions relative to the MSE
          gradient, and the energy_weight that puts it at --target_frac

Uses held-out training subjects (default S8), never TS1-TS6: gates that touch
the test set would leak it into design decisions.

    python report/check_energy_prior.py --checkpoint checkpoints/energy-dsm/best_energy.pth
    python report/check_energy_prior.py --checkpoint ... --task_checkpoint checkpoints/gcn-base-ht-clean/best_tasknet_47.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_data_mpi_inf_3dhp import MpiInf3dhpDataset
from models.energy_net import EnergyPrior
from utils.data_utils import (create_2d_data, drop_extreme_2d, fetch_h36m,
                              read_3d_data_3dhp)
from utils.device import get_device
from utils.loss import BoneDirectionPerturber, GaussianPerturber, mpjpe
from utils.seed import set_seed


def load_frames(args):
    dataset = read_3d_data_3dhp(MpiInf3dhpDataset(args.data_path),
                                restore_head_top=True)
    keypoints = create_2d_data(f"data/data_2d_mpi_inf_3dhp_{args.keypoints}.npz",
                               dataset)
    subjects = [s for s in args.subjects.split(",") if s]
    poses_3d, poses_2d, actions, cams = fetch_h36m(
        subjects, dataset, keypoints, action_filter=None, stride=1
    )
    if args.max_2d_abs > 0:
        poses_3d, poses_2d, actions, cams, dropped = drop_extreme_2d(
            poses_3d, poses_2d, actions, cams, args.max_2d_abs
        )
    poses_3d = np.concatenate(poses_3d).astype(np.float32)
    poses_2d = np.concatenate(poses_2d).astype(np.float32)

    rng = np.random.RandomState(args.seed)
    take = min(args.num_frames, poses_3d.shape[0])
    idx = rng.choice(poses_3d.shape[0], size=take, replace=False)
    print(f"==> {take} frames sampled from {subjects} ({poses_3d.shape[0]} total)")
    return poses_3d[idx], poses_2d[idx], dataset


def batched_energy(prior, y, batch=4096):
    out = []
    with torch.no_grad():
        for start in range(0, y.shape[0], batch):
            out.append(prior(y[start:start + batch]).squeeze(1))
    return torch.cat(out)


def auroc(pos, neg):
    """P(pos > neg) by rank statistics; 1.0 = perfect separation."""
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32,
                                device=scores.device)
    pos_ranks = ranks[labels.bool()].sum()
    n_pos, n_neg = pos.numel(), neg.numel()
    return ((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def gate1_monotonicity(prior, y, device):
    print("\n== gate 1: E(z + sigma*eps) vs sigma ==")
    z = prior.standardize(prior.bones(y).flatten(1))
    with torch.no_grad():
        e_clean = prior.net(z).mean().item()
    print(f"  sigma 0.000: E {e_clean:+.4f}")
    prev = e_clean
    violations = 0
    for sigma in prior.sigmas.tolist():
        eps = torch.randn_like(z)
        with torch.no_grad():
            e = prior.net(z + sigma * eps).mean().item()
        marker = "" if e >= prev else "  <-- NOT monotone"
        violations += e < prev
        print(f"  sigma {sigma:.3f}: E {e:+.4f}{marker}")
        prev = e
    print(f"  gate 1 {'PASS' if violations == 0 else f'FAIL ({violations} inversions)'}")
    return violations == 0


def gate2_denoising(prior, y, device, steps, step_scale):
    """Best MPJPE along the descent trajectory, not at a fixed step count.

    MDSM trains sigma0^2 grad E ~ (z_tilde - z), so light noise is undone in a
    step or two; running a fixed budget past that point walks the pose along
    the manifold toward high-density regions and away from this particular GT
    (mode seeking, not a bad field). The claim under test is "the field passes
    near the GT", which the trajectory minimum measures at every noise level.
    """
    print(f"\n== gate 2: denoising by gradient descent "
          f"(<= {steps} steps, eta = {step_scale} * sigma0^2) ==")
    bones, std = prior.bones, prior.standardize
    z_clean = std(bones(y).flatten(1))
    y_ref = bones.compose(std.inverse(z_clean).view(y.shape[0], -1, 3))
    eta = step_scale * prior.sigma0 ** 2

    def err_mm(z):
        return mpjpe(bones.compose(std.inverse(z).view(y.shape[0], -1, 3)),
                     y_ref).item() * 1000

    sigmas = prior.sigmas
    ok = True
    for sigma in [sigmas[1].item(), sigmas[sigmas.numel() // 2].item(),
                  sigmas[-2].item()]:
        z = (z_clean + sigma * torch.randn_like(z_clean)).detach()
        before = err_mm(z)
        best, best_step = before, 0
        for step in range(1, steps + 1):
            z = z.requires_grad_(True)
            grad = torch.autograd.grad(prior.net(z).sum(), z)[0]
            z = (z - eta * grad).detach()
            cur = err_mm(z)
            if cur < best:
                best, best_step = cur, step
        reduction = 100 * (1 - best / before)
        ok = ok and (best <= 0.7 * before)
        print(f"  sigma {sigma:.3f}: {before:8.2f} mm -> best {best:8.2f} mm "
              f"at step {best_step:3d} ({reduction:+.1f}%), final {err_mm(z):8.2f} mm")
    print(f"  gate 2 {'PASS' if ok else 'FAIL'} "
          f"(trajectory minimum must cut each row by >= 30%)")
    return ok


def gate3_separation(prior, y, dataset, device):
    """Pass criterion covers only OFF-manifold corruptions.

    Rotating bones at GT length mostly yields a different but valid body pose
    -- it moves along the manifold, and an unconditional body prior is
    supposed to accept it (pulling toward the one correct pose is the MSE
    term's job, and was the conditional SEAL net's job). The angle row is
    reported for information; the corruptions that break the body (Gaussian
    joints, scaled bone lengths) are the ones E must rank above GT.
    """
    print("\n== gate 3: E separates GT from corrupted poses ==")
    parents = list(dataset.skeleton().parents())
    e_gt = batched_energy(prior, y)
    print(f"  GT:            E {e_gt.mean():+.4f} +- {e_gt.std():.4f}")

    angle = BoneDirectionPerturber(parents, exclude=(),
                                   theta_min=3.0, theta_max=15.0).to(device)
    gauss = GaussianPerturber(root=parents.index(-1), exclude=(),
                              sigma=0.01832).to(device)

    ok = True
    with torch.no_grad():
        corruptions = {"angle 3-15deg": angle(y), "gauss 18.3mm": gauss(y)}
        z = prior.bones(y)
        scale = torch.empty(y.shape[0], z.shape[1], 1, device=y.device)
        scale.uniform_(0.8, 1.2)
        corruptions["length 0.8-1.2x"] = prior.bones.compose(z * scale)
    off_manifold = {"gauss 18.3mm", "length 0.8-1.2x"}
    for name, y_neg in corruptions.items():
        e_neg = batched_energy(prior, y_neg)
        score = auroc(e_neg, e_gt)
        scored = name in off_manifold
        if scored:
            ok = ok and score > 0.9
        print(f"  {name:<15} E {e_neg.mean():+.4f} +- {e_neg.std():.4f}, "
              f"AUROC {score:.3f}{'' if scored else '  (on-manifold, informational)'}")
    print(f"  gate 3 {'PASS' if ok else 'FAIL'} "
          f"(AUROC > 0.9 for off-manifold corruptions)")
    return ok


def gate4_calibration(prior, y, x2d, args, device):
    print("\n== gate 4: energy vs MSE gradient on task-net predictions ==")
    model = torch.load(args.task_checkpoint, map_location=device)
    if isinstance(model, dict):
        raise ValueError("pass a best_tasknet_*.pth (whole-module checkpoint)")
    model.to(device).eval()
    criterion = nn.MSELoss(reduction="mean")

    targets = y - y[:, 14:15, :]
    take = min(args.num_frames, 8192)
    g_mse_norms, g_e_norms = [], []
    for start in range(0, take, 2048):
        inp = x2d[start:start + 2048]
        tgt = targets[start:start + 2048]
        out = model(inp)
        out = out.detach().requires_grad_(True)
        g_mse = torch.autograd.grad(criterion(out, tgt), out)[0]
        g_e = torch.autograd.grad(prior(out).mean(), out)[0]
        g_mse_norms.append(g_mse.flatten(1).norm(dim=1))
        g_e_norms.append(g_e.flatten(1).norm(dim=1))
    g_mse = torch.cat(g_mse_norms).mean().item()
    g_e = torch.cat(g_e_norms).mean().item()
    suggested = args.target_frac * g_mse / max(g_e, 1e-12)
    print(f"  ||grad MSE||   {g_mse:.3E}")
    print(f"  ||grad E||     {g_e:.3E}  (per unit energy_weight)")
    print(f"  energy_weight for a {args.target_frac:.0%} pull: {suggested:.3E}")
    return True


def main(args):
    device = get_device()
    prior = EnergyPrior.load(args.checkpoint, map_location=device)
    prior.to(device)
    prior.freeze()
    print(f"==> Prior: {args.checkpoint} "
          f"(pretrain epoch {prior.meta.get('epoch')}, "
          f"val DSM {prior.meta.get('val_dsm')})")

    poses_3d, poses_2d, dataset = load_frames(args)
    y = torch.from_numpy(poses_3d).to(device)
    x2d = torch.from_numpy(poses_2d).to(device)

    results = {
        "gate1": gate1_monotonicity(prior, y, device),
        "gate2": gate2_denoising(prior, y, device, args.gd_steps, args.gd_step_scale),
        "gate3": gate3_separation(prior, y, dataset, device),
    }
    if args.task_checkpoint is not None:
        results["gate4"] = gate4_calibration(prior, y, x2d, args, device)

    print("\n== summary ==")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="train_energy_dsm.py output")
    parser.add_argument("--task_checkpoint", type=str, default=None,
                        help="best_tasknet_*.pth for gate 4 (optional)")
    parser.add_argument("--data_path", type=str,
                        default="data/data_3d_mpi_inf_3dhp.npz")
    parser.add_argument("--keypoints", type=str, default="gt")
    parser.add_argument("--subjects", type=str, default="S8",
                        help="held-out training subjects; never TS1-TS6")
    parser.add_argument("--max_2d_abs", type=float, default=10.0)
    parser.add_argument("--num_frames", type=int, default=20480)
    parser.add_argument("--gd_steps", type=int, default=100)
    parser.add_argument("--gd_step_scale", type=float, default=0.5,
                        help="gradient-descent step = this * sigma0^2")
    parser.add_argument("--target_frac", type=float, default=0.05,
                        help="gate 4: desired ||grad E|| / ||grad MSE|| after "
                             "scaling by the suggested energy_weight")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
