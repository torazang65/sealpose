"""Unconditional DSM-trained energy prior over bone-vector coordinates.

The SEAL energy net learned E(x, y) from a hinge on energy *values* at the
finitely many (prediction, GT) pairs the run happened to visit, which leaves
the gradient field -- the only thing the task net ever consumes -- entirely
unconstrained (report/2026-08-19_seal_synthetic_negatives.md; the clean runs
of 2026-08-25 confirmed E-diff grows while MPJPE stays at baseline). This
module replaces that with an unconditional E(z) whose gradient field is the
regression target itself: multi-scale denoising score matching (MDSM, Li et
al. 2019) over bone vectors z = By.

Bone vectors rather than joint positions because the body prior is nearly
axis-aligned there -- "femur length is constant" is a constraint on one
3-vector's norm instead of a curved constraint tying six joint coordinates --
and because B is linear and invertible, so p_z and p_y differ by a constant
|det B| and Gaussian DSM in z-space is exact, no manifold corrections needed.
"""

import torch
import torch.nn as nn


class BoneTransform(nn.Module):
    """z = By: joint positions -> per-bone offset vectors, and back.

    The transform drops the root coordinate, so it is invariant to which joint
    the caller centred on (any translation cancels in the differences).
    compose() reassembles a pose from bone vectors with the root pinned at the
    origin; it inverts forward() up to that global translation.

    Args:
        parents: parent index per joint, -1 for the root. Requires the real
                 body skeleton -- on 3DHP that means --restore_head_top, since
                 the raw slot 0 stores the camera trajectory, not a joint.
    """

    def __init__(self, parents):
        super(BoneTransform, self).__init__()
        parents = [int(p) for p in parents]
        if parents.count(-1) != 1:
            raise ValueError(f"expected exactly one root, got parents={parents}")
        self.parents = parents
        self.num_joints = len(parents)
        bones = [j for j, p in enumerate(parents) if p != -1]
        index = {j: i for i, j in enumerate(bones)}

        # Forward-kinematics order for compose(): place a bone only once its
        # parent joint has been placed (same construction as
        # utils.loss.BoneDirectionPerturber).
        placed = {parents.index(-1)}
        order = []
        while len(order) < len(bones):
            progressed = False
            for j in bones:
                if j not in placed and parents[j] in placed:
                    order.append(j)
                    placed.add(j)
                    progressed = True
            if not progressed:
                raise ValueError("skeleton parents do not form a tree")
        self._fk = [(j, parents[j], index[j]) for j in order]

        self.register_buffer("child", torch.as_tensor(bones, dtype=torch.long))
        self.register_buffer("parent", torch.as_tensor([parents[j] for j in bones],
                                                       dtype=torch.long))

    @property
    def num_bones(self):
        return self.child.numel()

    def forward(self, y):
        """(B, J, 3) or (B, J*3) joint positions -> (B, n_bones, 3) bone vectors."""
        y = y.view(y.shape[0], -1, 3)
        return y[:, self.child] - y[:, self.parent]

    def compose(self, z):
        """(B, n_bones, 3) bone vectors -> (B, J, 3) pose with the root at the origin."""
        z = z.view(z.shape[0], -1, 3)
        out = z.new_zeros(z.shape[0], self.num_joints, 3)
        for joint, parent, i in self._fk:
            out[:, joint] = out[:, parent] + z[:, i]
        return out


class Standardizer(nn.Module):
    """Per-coordinate z-scoring with train-set statistics.

    Mandatory before DSM: bone-length axes have tiny variance (lengths are
    near-constant within a subject), so without this the score along those
    axes scales like 1/variance and dominates the landscape. After z-scoring a
    single sigma ladder applies uniformly to every coordinate.
    """

    def __init__(self, mean, std):
        super(Standardizer, self).__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-4))

    def forward(self, z_flat):
        return (z_flat - self.mean) / self.std

    def inverse(self, z_flat):
        return z_flat * self.std + self.mean


class EnergyNet(nn.Module):
    """Scalar energy over standardized bone vectors.

    Plain MLP, but with two hard requirements DSM adds over LinearLossNet:
    no dropout (a stochastic gradient field cannot be regressed) and a smooth
    activation (a piecewise-linear net has a piecewise-constant gradient in z,
    which cannot follow the continuous regression target). No batch norm for
    the same reason train/eval gradient fields must coincide.
    """

    def __init__(self, in_dim, hidden=512, depth=3):
        super(EnergyNet, self).__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.depth = depth
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        # No output bias: DSM supervises only grad E, which the final bias
        # never enters, so it would be a dead parameter pinning an arbitrary
        # additive constant.
        layers += [nn.Linear(hidden, 1, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_flat):
        return self.net(z_flat)


class MultiScaleDSM(nn.Module):
    """MDSM loss: sum_sigma l(sigma) E ||z - z~ + sigma_0^2 grad E(z~)||^2.

    One sigma per sample, drawn uniformly from the ladder, so every batch
    mixes scales. l(sigma) = 1/sigma^2: the residual magnitude grows like
    sigma, so the unweighted sum would be dominated by the largest scales and
    the near-manifold field -- the part the task net lives in -- would never
    train. sigma_0 is a single fixed anchor (the energy stays unconditional).

    Pass a CPU torch.Generator to make the noise deterministic -- the
    validation loss must not jitter between evals or patience-based early
    stopping triggers on noise instead of on the model.
    """

    def __init__(self, sigmas, sigma0):
        super(MultiScaleDSM, self).__init__()
        self.register_buffer("sigmas", torch.as_tensor(sigmas, dtype=torch.float32))
        self.sigma0 = float(sigma0)

    def forward(self, net, z_flat, generator=None, create_graph=True):
        """Returns (loss, sigma_idx, per_sample_loss); the last two detached."""
        batch, dim = z_flat.shape
        num_sigmas = self.sigmas.numel()
        if generator is not None:
            idx = torch.randint(num_sigmas, (batch,), generator=generator).to(z_flat.device)
            eps = torch.randn((batch, dim), generator=generator).to(z_flat.device)
        else:
            idx = torch.randint(num_sigmas, (batch,), device=z_flat.device)
            eps = torch.randn_like(z_flat)
        sigma = self.sigmas[idx].unsqueeze(1)

        z_tilde = (z_flat + sigma * eps).detach().requires_grad_(True)
        energy = net(z_tilde)
        grad = torch.autograd.grad(energy.sum(), z_tilde, create_graph=create_graph)[0]

        resid = z_flat.detach() - z_tilde + (self.sigma0 ** 2) * grad
        per_sample = resid.pow(2).sum(dim=1) / sigma.squeeze(1).pow(2)
        return per_sample.mean(), idx.detach(), per_sample.detach()


class EnergyPrior(nn.Module):
    """Frozen pretrained prior for task training (train_lifting --type dsm-prior).

    forward(y) chains bones -> standardize -> E, so the task net's gradient is
    B^T J_std^T grad_z E: each bone's gradient lands on its two endpoint
    joints with opposite signs, spring-force style. The wrapper carries the
    train-time standardizer statistics so task training cannot drift from the
    pretraining coordinates.
    """

    def __init__(self, parents, mean, std, hidden=512, depth=3, sigmas=None, sigma0=0.1):
        super(EnergyPrior, self).__init__()
        self.bones = BoneTransform(parents)
        self.standardize = Standardizer(mean, std)
        self.net = EnergyNet(self.bones.num_bones * 3, hidden=hidden, depth=depth)
        self.register_buffer(
            "sigmas",
            torch.as_tensor(sigmas if sigmas is not None else [], dtype=torch.float32),
        )
        self.sigma0 = float(sigma0)
        self.meta = {}

    def forward(self, y):
        z = self.bones(y).flatten(1)
        return self.net(self.standardize(z))

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        if ckpt.get("kind") != "dsm_energy_prior":
            raise ValueError(
                f"{path} is not a DSM energy-prior checkpoint "
                f"(kind={ckpt.get('kind')!r}); it cannot be used with --type dsm-prior"
            )
        prior = cls(
            parents=ckpt["parents"],
            mean=ckpt["mean"],
            std=ckpt["std"],
            hidden=ckpt["hidden"],
            depth=ckpt["depth"],
            sigmas=ckpt["sigmas"],
            sigma0=ckpt["sigma0"],
        )
        prior.net.load_state_dict(ckpt["net"])
        prior.meta = {k: ckpt[k] for k in ("train_args", "epoch", "val_dsm") if k in ckpt}
        return prior


def save_energy_checkpoint(path, net, parents, mean, std, sigmas, sigma0,
                           train_args=None, epoch=None, val_dsm=None):
    """Single writer for the checkpoint format EnergyPrior.load expects."""
    torch.save(
        {
            "kind": "dsm_energy_prior",
            "parents": [int(p) for p in parents],
            "hidden": net.hidden,
            "depth": net.depth,
            "mean": torch.as_tensor(mean, dtype=torch.float32).flatten().cpu(),
            "std": torch.as_tensor(std, dtype=torch.float32).flatten().cpu(),
            "sigmas": torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu(),
            "sigma0": float(sigma0),
            "net": {k: v.cpu() for k, v in net.state_dict().items()},
            "train_args": train_args,
            "epoch": epoch,
            "val_dsm": val_dsm,
        },
        path,
    )
