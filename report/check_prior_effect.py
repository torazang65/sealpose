#!/usr/bin/env python
"""Did the frozen DSM prior actually move the task net?

Gate 4 sized energy_weight from the baseline's operating point before the run.
This measures what the term did during it, on the TEST split the results are
reported on, along three axes:

  A  force ratio   ||lambda*grad E|| / ||grad MSE|| on each net's own
                   predictions -- how hard the prior pulled, in the units the
                   optimizer actually saw
  B  alignment     cos(-grad E, y_GT - y_hat) -- whether that pull pointed at
                   the ground truth, at chance (0.0) or against it
  C  effect        E(y_hat) for the dsm-prior net vs the baseline net, both
                   against E(y_GT). If the prior worked, its predictions sit
                   at lower energy than the baseline's; if the two agree, the
                   term never changed the solution

Also reports the same numbers at 3x the run's lambda, to separate "the prior
is useless here" from "the prior was scaled too weakly to matter".

    python report/check_prior_effect.py \
        --prior checkpoints/energy-dsm-smin005/best_energy.pth \
        --dsm_checkpoint checkpoints/gcn-dsmprior/best_tasknet_38.pth \
        --base_checkpoint checkpoints/gcn-base-ht-clean/best_tasknet_47.pth \
        --energy_weight 2.745e-7
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_data_mpi_inf_3dhp import MpiInf3dhpDataset
from data_loader import PoseDataSet
from models.energy_net import EnergyPrior
from utils.data_utils import create_2d_data, fetch_h36m, read_3d_data_3dhp
from utils.device import get_device
from utils.seed import set_seed

BASE_JOINT = 14
TEST_SUBJECTS = ["TS1", "TS2", "TS3", "TS4"]


def test_loader(args, batch_size=1024):
    dataset = read_3d_data_3dhp(MpiInf3dhpDataset(args.data_path),
                                restore_head_top=True)
    keypoints = create_2d_data(f"data/data_2d_mpi_inf_3dhp_{args.keypoints}.npz",
                               dataset)
    poses_3d, poses_2d, actions, cams = fetch_h36m(
        TEST_SUBJECTS, dataset, keypoints, action_filter=None, stride=1
    )
    return DataLoader(PoseDataSet(poses_3d, poses_2d, actions, cams),
                      batch_size=batch_size, shuffle=False, num_workers=0)


def load_task_net(path, device):
    model = torch.load(path, map_location=device)
    if isinstance(model, dict):
        raise ValueError(f"{path}: pass a whole-module best_tasknet_*.pth")
    return model.to(device).eval()


def measure(model, prior, loader, lam, device, max_batches=0):
    """Per-sample force ratio, alignment, energies and MPJPE for one task net."""
    criterion = nn.MSELoss(reduction="mean")
    ratios, cosines, grad_cos, e_hat, e_gt, errs = [], [], [], [], [], []
    for i, batch in enumerate(loader):
        targets, inputs = batch[0].to(device), batch[1].to(device)
        targets = targets - targets[:, BASE_JOINT:BASE_JOINT + 1, :]
        with torch.no_grad():
            out = model(inputs)
        out = out.detach().requires_grad_(True)

        g_mse = torch.autograd.grad(criterion(out, targets), out)[0]
        g_e = torch.autograd.grad(prior(out).mean(), out)[0]

        # Per-sample norms; the loss means divide by batch size, so both
        # gradients carry the same 1/B factor and the ratio is unaffected.
        n_mse = g_mse.flatten(1).norm(dim=1)
        n_e = (lam * g_e).flatten(1).norm(dim=1)
        ratios.append((n_e / n_mse.clamp_min(1e-30)).detach().cpu())

        # Does -grad E point from the prediction toward the ground truth?
        # With pos_loss=mse this is identical to cos(grad E, grad MSE):
        # grad MSE = (2/N)(y_hat - y_GT), so -grad MSE is parallel to to_gt and
        # cos(-a, -b) = cos(a, b). cos_gg re-measures it the other way as a
        # check that the two really do coincide.
        to_gt = (targets - out).flatten(1)
        neg_ge = (-g_e).flatten(1)
        cos = torch.nn.functional.cosine_similarity(neg_ge, to_gt, dim=1)
        cosines.append(cos.detach().cpu())
        cos_gg = torch.nn.functional.cosine_similarity(
            g_e.flatten(1), g_mse.flatten(1), dim=1
        )
        grad_cos.append(cos_gg.detach().cpu())

        with torch.no_grad():
            e_hat.append(prior(out).squeeze(1).cpu())
            e_gt.append(prior(targets).squeeze(1).cpu())
            # eval.py re-centres the prediction on the base joint before
            # scoring; match it so this column is the reported Protocol #1.
            # The gradients above deliberately do not, since the optimizer saw
            # the raw output. Energies are unaffected either way -- bone
            # vectors are translation invariant.
            centred = out.detach() - out.detach()[:, BASE_JOINT:BASE_JOINT + 1, :]
            errs.append((centred - targets).norm(dim=-1).mean(dim=1).cpu() * 1000)
        if max_batches and i + 1 >= max_batches:
            break
    return (torch.cat(ratios), torch.cat(cosines), torch.cat(e_hat),
            torch.cat(e_gt), torch.cat(errs), torch.cat(grad_cos))


def main(args):
    device = get_device()
    prior = EnergyPrior.load(args.prior, map_location=device).to(device).freeze()
    loader = test_loader(args)
    print(f"==> Prior: {args.prior} (pretrain epoch {prior.meta.get('epoch')})")
    print(f"==> Test split: {TEST_SUBJECTS}, lambda = {args.energy_weight:.3E}")

    nets = {"dsm-prior": args.dsm_checkpoint, "baseline": args.base_checkpoint}
    out = {}
    for name, path in nets.items():
        if path is None:
            continue
        model = load_task_net(path, device)
        out[name] = measure(model, prior, loader, args.energy_weight, device,
                            args.max_batches)

    print("\n== A. force ratio  ||lambda*grad E|| / ||grad MSE|| ==")
    for name, (ratio, _, _, _, _, _) in out.items():
        q = np.percentile(ratio.numpy(), [10, 50, 90])
        print(f"  {name:<10} median {q[1]:.4f}  (p10 {q[0]:.4f}, p90 {q[2]:.4f})")
    print(f"  at 3x lambda, median would be "
          f"{3 * np.median(out['dsm-prior'][0].numpy()):.4f}")

    print("\n== B. alignment  cos(-grad E, y_GT - y_hat) ==")
    for name, (_, cos, _, _, _, _) in out.items():
        q = np.percentile(cos.numpy(), [10, 50, 90])
        frac = (cos > 0).float().mean().item()
        print(f"  {name:<10} median {q[1]:+.4f}  (p10 {q[0]:+.4f}, p90 {q[2]:+.4f}), "
              f"{frac:.1%} of samples point toward GT")

    for name, (_, cos, _, _, _, gcos) in out.items():
        print(f"  {name:<10} cos(grad E, grad MSE) median {np.median(gcos.numpy()):+.4f} "
              f"-- identical to the row above when pos_loss=mse, "
              f"max |diff| {np.abs(cos.numpy() - gcos.numpy()).max():.2e}")

    print("\n== C. effect  E(y_hat) vs E(y_GT) ==")
    for name, (_, _, eh, eg, err, _) in out.items():
        print(f"  {name:<10} E(y_hat) {eh.mean():+9.3f} +- {eh.std():6.3f}, "
              f"E-diff {(eh - eg).mean():+8.3f}, MPJPE {err.mean():.2f} mm")
    if len(out) == 2:
        d = (out["dsm-prior"][2].mean() - out["baseline"][2].mean()).item()
        gt = out["dsm-prior"][3].mean().item()
        gap = (out["baseline"][2].mean() - out["baseline"][3].mean()).item()
        print(f"  E(y_GT) {gt:+.3f}; baseline sits {gap:+.3f} above it")
        print(f"  dsm-prior minus baseline: {d:+.3f} "
              f"({100 * d / gap if gap else float('nan'):+.1f}% of that gap closed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=str, required=True)
    parser.add_argument("--dsm_checkpoint", type=str, required=True)
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--energy_weight", type=float, default=2.745e-7)
    parser.add_argument("--data_path", type=str,
                        default="data/data_3d_mpi_inf_3dhp.npz")
    parser.add_argument("--keypoints", type=str, default="gt")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
