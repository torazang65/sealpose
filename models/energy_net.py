"""Unconditional DSM-trained energy prior over the pose.

The SEAL energy net learned E(x, y) from a hinge on energy *values* at the
finitely many (prediction, GT) pairs the run happened to visit, which leaves
the gradient field -- the only thing the task net ever consumes -- entirely
unconstrained (report/2026-08-19_seal_synthetic_negatives.md; the clean runs
of 2026-08-25 confirmed E-diff grows while MPJPE stays at baseline). This
module replaces that with an unconditional energy whose gradient field is the
regression target itself: multi-scale denoising score matching (MDSM, Li et
al. 2019).

Two input representations, selected per checkpoint by `features`:

  "joint"  x = root-centred positions of the non-root joints, in metres, no
           standardisation. Noise is added to these same coordinates, so the
           corruption, the regression target sigma0^2 grad E ~ x~ - x, and
           every diagnostic live in one space with a physical unit -- the
           space the task net's residual lives in. This is the control the
           later representations are measured against: whether a DSM scalar
           energy learns at all, before any transform is put in front of it.

  "bone"   z = By, per-bone offset vectors, z-scored per coordinate; noise is
           added in the standardised space. B is linear and invertible, so
           p_z and p_y differ by the constant |det B| and Gaussian DSM in
           z-space is exact. The representation of the 2026-08-26 report,
           kept so that report reproduces.

  "scalebone"  E(z(x)) with z_i = b_i / s(x), b = Bx the bone vectors of the
           root-centred pose and s(x) = mean_i |b_i| the mean bone length.
           Noise is still added to the joint coordinates x, exactly as for
           "joint" (same ladder, same metres), so the corruption and the
           regression target stay in the task net's space; only the net's
           input changes, and the DSM gradient reaches x through the chain
           rule, grad_x E = J_z^T grad_z E. z is invariant to translation and
           to a uniform scaling of the pose (rotation and bone proportions
           survive), so uniform scaling is a null direction of grad_x E: the
           prior can push the task net's pose toward a body-like *shape* but
           never toward a body *size*. Motivation: the joint prior's -grad E
           was spread uniformly over joints and fought the MSE (report
           2026-09-22); removing the size degree of freedom removes one way
           for the two to disagree.
"""

import torch
import torch.nn as nn


class RootCentered(nn.Module):
    """x = (y - y_root) over the non-root joints, and back.

    Same interface as BoneTransform: forward() drops the root (which is zero
    after centring), compose() puts it back at the origin. Since the root
    slot is removed rather than pinned, no noise ever lands on it. Through
    forward() the energy is translation invariant, so its gradient w.r.t. the
    full pose sums to zero over joints: the root receives minus the sum of
    the others, exactly as B^T does for bone vectors.

    Args:
        parents: parent index per joint, -1 for the root.
    """

    def __init__(self, parents):
        super(RootCentered, self).__init__()
        parents = [int(p) for p in parents]
        if parents.count(-1) != 1:
            raise ValueError(f"expected exactly one root, got parents={parents}")
        self.parents = parents
        self.root = parents.index(-1)
        self.num_joints = len(parents)
        self.register_buffer("free", torch.as_tensor(
            [j for j in range(self.num_joints) if j != self.root], dtype=torch.long))

    @property
    def dim(self):
        return self.free.numel() * 3

    def forward(self, y):
        """(B, J, 3) or (B, J*3) joint positions -> (B, (J-1)*3) root-centred."""
        y = y.view(y.shape[0], -1, 3)
        return (y - y[:, self.root:self.root + 1])[:, self.free].flatten(1)

    def compose(self, x):
        """(B, (J-1)*3) -> (B, J, 3) pose with the root at the origin."""
        out = x.new_zeros(x.shape[0], self.num_joints, 3)
        out[:, self.free] = x.view(x.shape[0], -1, 3)
        return out


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
        self.root = parents.index(-1)
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


class ScaledBones(nn.Module):
    """z_i = b_i / s(x): bone vectors of a root-centred pose over their mean length.

    Input is the "joint" corruption space (root-centred non-root joints,
    flattened), so it composes after RootCentered and the noise never sees
    it. No standardizer: dividing by s(x) already makes every bone O(1), and
    a fixed per-coordinate z-score would put the scale back in. Not
    invertible (s is lost by construction), so it has no compose(); decoding
    goes through RootCentered, whose space the noise lives in.
    """

    def __init__(self, parents, eps=1e-6):
        super(ScaledBones, self).__init__()
        self.centred = RootCentered(parents)
        self.bones = BoneTransform(parents)
        self.eps = float(eps)

    @property
    def dim(self):
        return self.bones.num_bones * 3

    def forward(self, x_flat):
        """(B, (J-1)*3) root-centred -> (B, n_bones*3) scale-normalised bones."""
        b = self.bones(self.centred.compose(x_flat))
        s = b.norm(dim=2).mean(dim=1, keepdim=True).clamp_min(self.eps)
        return (b / s.unsqueeze(2)).flatten(1)

    def scale(self, x_flat):
        """s(x), the mean bone length in metres, for diagnostics."""
        b = self.bones(self.centred.compose(x_flat))
        return b.norm(dim=2).mean(dim=1)


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
    """Scalar energy over the flattened representation.

    Plain MLP, but with two hard requirements DSM adds over LinearLossNet:
    no dropout (a stochastic gradient field cannot be regressed) and a smooth
    activation (a piecewise-linear net has a piecewise-constant gradient,
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
    """MDSM loss: sum_sigma l(sigma) E ||x - x~ + sigma_0^2 grad E(x~)||^2.

    Generic in the corrupted variable: `x` is whatever space the noise lives
    in and `energy_fn` maps it to a scalar energy (the net itself for "joint"
    and "bone", where the net's input is the corruption space).

    One sigma per sample, drawn uniformly from the ladder, so every batch
    mixes scales. l(sigma) = 1/sigma^2: the residual magnitude grows like
    sigma, so the unweighted sum would be dominated by the largest scales and
    the near-manifold field -- the part the task net lives in -- would never
    train. sigma_0 is a single fixed anchor (the energy stays unconditional);
    the net never sees sigma, so it learns one field sigma0^2 grad E ~ x~ - x
    for the whole mixture, and one step of exactly that size is the natural
    denoiser -- which is what the `one_step` diagnostic measures.

    Pass a CPU torch.Generator to make the noise deterministic -- the
    validation loss must not jitter between evals or patience-based early
    stopping triggers on noise instead of on the model.
    """

    def __init__(self, sigmas, sigma0):
        super(MultiScaleDSM, self).__init__()
        self.register_buffer("sigmas", torch.as_tensor(sigmas, dtype=torch.float32))
        self.sigma0 = float(sigma0)

    def forward(self, energy_fn, x, generator=None, create_graph=True):
        """Returns (loss, sigma_idx, diagnostics), the last two detached.

        diagnostics, all per sample:
          "loss"      the weighted DSM residual, mean of which is the loss
          "cos"       cos(sigma0^2 grad E(x~), x~ - x): does the field point
                      back along the corruption
          "one_step"  ||x~ - sigma0^2 grad E - x|| / ||x~ - x||: error left
                      after the one denoising step the objective trains for
                      (< 1 is an improvement, 0 is perfect)
        """
        batch, dim = x.shape
        num_sigmas = self.sigmas.numel()
        if generator is not None:
            idx = torch.randint(num_sigmas, (batch,), generator=generator).to(x.device)
            eps = torch.randn((batch, dim), generator=generator).to(x.device)
        else:
            idx = torch.randint(num_sigmas, (batch,), device=x.device)
            eps = torch.randn_like(x)
        sigma = self.sigmas[idx].unsqueeze(1)

        x_tilde = (x + sigma * eps).detach().requires_grad_(True)
        energy = energy_fn(x_tilde)
        grad = torch.autograd.grad(energy.sum(), x_tilde, create_graph=create_graph)[0]

        step = (self.sigma0 ** 2) * grad
        resid = x.detach() - x_tilde + step
        per_sample = resid.pow(2).sum(dim=1) / sigma.squeeze(1).pow(2)

        with torch.no_grad():
            noise = (x_tilde - x).detach()
            noise_norm = noise.norm(dim=1).clamp_min(1e-12)
            cos = (step * noise).sum(dim=1) / (step.norm(dim=1).clamp_min(1e-12) * noise_norm)
            one_step = resid.norm(dim=1) / noise_norm
        diagnostics = {"loss": per_sample.detach(), "cos": cos.detach(),
                       "one_step": one_step.detach()}
        return per_sample.mean(), idx.detach(), diagnostics


class EnergyPrior(nn.Module):
    """Frozen pretrained prior for task training (train_lifting --type dsm-prior).

    forward(y) maps a pose to its energy through the representation the
    checkpoint was trained with:

      "joint"  E(x),                x = root-centred non-root joints, metres.
               The task net's gradient is grad_x E on the non-root joints
               and minus their sum on the root (translation null direction).
      "bone"   E(std(By)),          the task net's gradient is B^T J_std^T
               grad_z E: each bone's gradient lands on its two endpoint
               joints with opposite signs, spring-force style.
      "scalebone"  E(Bx / s(x)),    x as for "joint"; the gradient is the
               spring force of "bone" minus its projection on the uniform
               scaling direction (the s(x) term of the chain rule).

    corrupt_space / energy_at / decode expose the space the noise was added
    to, so gates and diagnostics probe the field where it was trained
    whichever representation is in use. For "scalebone" that space is the
    joint space and energy_at applies the bone encoding itself. The wrapper
    carries the train-time standardizer statistics (bone only) so task
    training cannot drift from the pretraining coordinates.
    """

    def __init__(self, parents, hidden=512, depth=3, sigmas=None, sigma0=0.1,
                 features="joint", mean=None, std=None):
        super(EnergyPrior, self).__init__()
        if features not in ("joint", "bone", "scalebone"):
            raise ValueError(f"unknown features={features!r}")
        self.features = features
        self.encode = None
        if features == "joint":
            self.transform = RootCentered(parents)
            self.standardize = None
            in_dim = self.transform.dim
        elif features == "scalebone":
            self.transform = RootCentered(parents)
            self.standardize = None
            self.encode = ScaledBones(parents)
            in_dim = self.encode.dim
        else:
            if mean is None or std is None:
                raise ValueError('features="bone" needs the standardizer mean/std')
            self.transform = BoneTransform(parents)
            self.standardize = Standardizer(mean, std)
            in_dim = self.transform.num_bones * 3
        self.net = EnergyNet(in_dim, hidden=hidden, depth=depth)
        self.register_buffer(
            "sigmas",
            torch.as_tensor(sigmas if sigmas is not None else [], dtype=torch.float32),
        )
        self.sigma0 = float(sigma0)
        self.meta = {}

    @property
    def parents(self):
        return self.transform.parents

    def corrupt_space(self, y):
        """Pose -> the coordinates pretraining added its noise to (= net input)."""
        x = self.transform(y).flatten(1)
        return x if self.standardize is None else self.standardize(x)

    def energy_at(self, x):
        """Energy of a point given in the corruption space."""
        return self.net(x if self.encode is None else self.encode(x))

    def decode(self, x):
        """Inverse of corrupt_space, up to translation (root at the origin)."""
        if self.standardize is not None:
            x = self.standardize.inverse(x)
        return self.transform.compose(x)

    def forward(self, y):
        return self.energy_at(self.corrupt_space(y))

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
            hidden=ckpt["hidden"],
            depth=ckpt["depth"],
            sigmas=ckpt["sigmas"],
            sigma0=ckpt["sigma0"],
            # Absent in checkpoints written before the joint representation
            # existed; those are all bone-vector priors.
            features=ckpt.get("features", "bone"),
            mean=ckpt.get("mean"),
            std=ckpt.get("std"),
        )
        prior.net.load_state_dict(ckpt["net"])
        prior.meta = {k: ckpt[k] for k in ("train_args", "epoch", "val_dsm") if k in ckpt}
        return prior


def save_energy_checkpoint(path, net, parents, sigmas, sigma0, features,
                           mean=None, std=None, train_args=None, epoch=None,
                           val_dsm=None):
    """Single writer for the checkpoint format EnergyPrior.load expects."""
    if features == "bone" and (mean is None or std is None):
        raise ValueError('features="bone" needs the standardizer mean/std')
    torch.save(
        {
            "kind": "dsm_energy_prior",
            "features": features,
            "parents": [int(p) for p in parents],
            "hidden": net.hidden,
            "depth": net.depth,
            "mean": None if mean is None else torch.as_tensor(mean, dtype=torch.float32).flatten().cpu(),
            "std": None if std is None else torch.as_tensor(std, dtype=torch.float32).flatten().cpu(),
            "sigmas": torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu(),
            "sigma0": float(sigma0),
            "net": {k: v.cpu() for k, v in net.state_dict().items()},
            "train_args": train_args,
            "epoch": epoch,
            "val_dsm": val_dsm,
        },
        path,
    )
