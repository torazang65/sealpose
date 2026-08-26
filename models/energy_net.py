"""Unconditional DSM-trained energy prior over the body's shape coordinates.

The SEAL energy net learned E(x, y) from a hinge on energy *values* at the
finitely many (prediction, GT) pairs the run happened to visit, which leaves
the gradient field -- the only thing the task net ever consumes -- entirely
unconstrained (report/2026-08-19_seal_synthetic_negatives.md). This module
replaces that with an unconditional energy whose gradient field is the
regression target itself: multi-scale denoising score matching (MDSM, Li et
al. 2019).

Two input representations, selected per checkpoint by `features`:

  "bone"       z = By, the raw per-bone offset vectors. B is linear and
               invertible, so p_z and p_y differ by the constant |det B| and
               Gaussian DSM in z-space is exact. This is the representation
               of the 2026-08-26 report.

  "invariant"  (log bone length / geometric mean, cosines of adjacent bone
               pairs) -- invariant to translation, global rotation AND global
               scale. Motivated by that report: the bone-vector prior spent
               capacity on absolute stature and camera orientation, neither of
               which transfers to a new subject, and its structural gains were
               confined to left-right symmetry (the one property every skeleton
               shares). Proportions are far more universal across people than
               absolute size, so quotienting scale out keeps the transferable
               part and drops the rest.

For "invariant" the corruption stays in GEOMETRY space and never in feature
space: y~ = y + sigma*eps over the non-root joints ("joint", the default), or
b~ = b + sigma*eps over the bone vectors ("bone"). Both keep every corrupted
sample a real configuration, and both are 48-dimensional -- the root is left
out so no noise lands on a translation the energy cannot see.

They are NOT interchangeable. B is invertible but not orthogonal, so isotropic
noise in one space is anisotropic in the other: perturbing bone vectors
uniformly makes joint displacement accumulate along the kinematic chain (root
0 mm, wrist 2.26x the hips), while perturbing joints uniformly makes every
joint equally wrong, which is what a lifting network's error actually looks
like.

Corrupting the features instead would be wrong for two reasons, the second the
deeper one: cos in [-1, 1] means Gaussian noise leaves the feature domain, and,
more fundamentally, the length and angle blocks are geometrically coupled
through the pose, so independent per-block noise produces a feature vector no
real skeleton can realise. Perturbing the geometry lets both blocks move
together the way that coupling dictates. Invariance is then a structural
constraint on the energy, not a property of the noise.
"""

from collections import defaultdict

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
        self.root = parents.index(-1)
        self.num_joints = len(parents)
        bones = [j for j, p in enumerate(parents) if p != -1]
        index = {j: i for i, j in enumerate(bones)}

        # Forward-kinematics order for compose(): place a bone only once its
        # parent joint has been placed (same construction as
        # utils.loss.BoneDirectionPerturber).
        placed = {self.root}
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


class InvariantFeatures(nn.Module):
    """Bone vectors -> similarity-invariant shape coordinates.

    Emits the angle block always, and a length block chosen by `lengths`:

        "full"      log(||b_i|| / s), one per bone, s the geometric mean
        "symmetry"  log(||b_left|| / ||b_right||), one per mirrored pair
        "none"      no length features at all
        cos(b_i, b_j)   one per pair of bones sharing a joint (always)

    The length block is where subject identity hides. 3DHP's bones are rigid
    within a subject -- per-bone length std is 2e-6 m across all of S1 -- so
    the 16 log-lengths take only as many distinct values as there are
    subjects. Seven training subjects means an effective sample size of seven
    for a 15-dimensional distribution, however many million frames the loader
    reports, and "full" duly memorises them: the smallest sigma rung's
    validation loss climbs from the first epoch whatever that rung is set to.

    "symmetry" keeps the part that is learnable from seven subjects, because
    it is the same constraint in every one of them: left and right bones match.
    That is also the only structural property the 2026-08-26 bone-vector run
    ever improved (LSE), while absolute length never moved. The remaining
    proportion dimensions -- torso against limb and so on -- genuinely differ
    per person and seven samples cannot describe them.

    "full" makes the length block sum to zero exactly, one linear dependency
    among the features; J_T is rank deficient either way, so anything
    inverting J J^T needs a pseudo-inverse rather than a small ridge.

    Adjacent pairs cover both bones meeting head to tail and siblings hanging
    off the same joint. This deliberately omits dihedrals, so the map is NOT
    complete: bone lengths plus adjacent angles leave the torsion about each
    bone free, and cosines are invariant to reflection, so chirality is lost
    too. On the 3DHP skeleton it emits 16 + 19 = 35 numbers for 44 shape
    degrees of freedom (48 bone coordinates less 3 rotation and 1 scale), and
    report/check_energy_prior.py measures what that costs as the fraction of a
    perturbation the features cannot see. Add signed dihedrals if it is large.

    Args:
        parents:      parent index per joint, -1 for the root.
        lengths:      "full" | "symmetry" | "none", see above.
        joints_left,
        joints_right: mirrored joint lists, required for "symmetry". A bone is
                      named by its child joint, so these pair the bones too.
        eps:          floor on bone length, metres. A pose perturbed at the top
                      of the sigma ladder can collapse a bone toward zero,
                      where the log and the unit vector would both blow up.
    """

    def __init__(self, parents, lengths="symmetry", joints_left=None,
                 joints_right=None, eps=1e-3):
        super(InvariantFeatures, self).__init__()
        parents = [int(p) for p in parents]
        bones = [j for j, p in enumerate(parents) if p != -1]
        slot = {j: i for i, j in enumerate(bones)}

        if lengths not in ("full", "symmetry", "none"):
            raise ValueError(f"unknown lengths={lengths!r}")
        self.lengths = lengths
        left_slot, right_slot = [], []
        if lengths == "symmetry":
            if joints_left is None or joints_right is None:
                raise ValueError('lengths="symmetry" needs joints_left/right')
            for l, r in zip(joints_left, joints_right):
                if int(l) in slot and int(r) in slot:
                    left_slot.append(slot[int(l)])
                    right_slot.append(slot[int(r)])
        self.register_buffer("left", torch.as_tensor(left_slot, dtype=torch.long))
        self.register_buffer("right", torch.as_tensor(right_slot, dtype=torch.long))

        incident = defaultdict(list)
        for j in bones:
            incident[j].append(slot[j])              # bone j ends at joint j
            incident[parents[j]].append(slot[j])     # and starts at its parent
        pairs = set()
        for bs in incident.values():
            for a in range(len(bs)):
                for b in range(a + 1, len(bs)):
                    pairs.add((min(bs[a], bs[b]), max(bs[a], bs[b])))
        pairs = sorted(pairs)

        self.num_bones = len(bones)
        self.num_pairs = len(pairs)
        self.num_lengths = {"full": len(bones), "symmetry": len(left_slot),
                            "none": 0}[lengths]
        self.eps = eps
        self.register_buffer("pair_i", torch.as_tensor([p[0] for p in pairs],
                                                       dtype=torch.long))
        self.register_buffer("pair_j", torch.as_tensor([p[1] for p in pairs],
                                                       dtype=torch.long))

    @property
    def dim(self):
        return self.num_lengths + self.num_pairs

    def forward(self, b):
        """(B, n_bones, 3) bone vectors -> (B, num_lengths + n_pairs)."""
        b = b.view(b.shape[0], -1, 3)
        length = b.norm(dim=-1).clamp_min(self.eps)              # (B, n_bones)
        log_length = length.log()

        u = b / length.unsqueeze(-1)
        z_angle = (u[:, self.pair_i] * u[:, self.pair_j]).sum(-1)  # (B, n_pairs)
        if self.lengths == "none":
            return z_angle
        if self.lengths == "symmetry":
            z_length = log_length[:, self.left] - log_length[:, self.right]
        else:
            z_length = log_length - log_length.mean(dim=1, keepdim=True)
        return torch.cat([z_length, z_angle], dim=1)


class Standardizer(nn.Module):
    """Per-coordinate z-scoring with train-set statistics.

    Mandatory before DSM: the coordinates have wildly different spreads --
    bone-length axes are near-constant within a subject, and in the invariant
    representation log-proportions vary far less than cosines -- so without
    this the score along the tight axes scales like 1/variance and dominates
    the landscape.
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
    """Scalar energy over standardized coordinates.

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

    Generic in the corrupted variable: pass whatever space the noise lives in
    as `x` and a callable mapping it to a scalar energy. For features="bone"
    that is standardized bone vectors and the callable is the net itself; for
    features="invariant" it is the root-centred pose in metres and the
    callable runs bones -> features -> standardize -> net.

    One sigma per sample, drawn uniformly from the ladder, so every batch
    mixes scales. l(sigma) = 1/sigma^2: the residual magnitude grows like
    sigma, so the unweighted sum would be dominated by the largest scales and
    the near-manifold field -- the part the task net lives in -- would never
    train. sigma_0 is a single fixed anchor (the energy stays unconditional);
    it only rescales E, and energy_weight is recalibrated against it anyway.

    Pass a CPU torch.Generator to make the noise deterministic -- the
    validation loss must not jitter between evals or patience-based early
    stopping triggers on noise instead of on the model.
    """

    def __init__(self, sigmas, sigma0):
        super(MultiScaleDSM, self).__init__()
        self.register_buffer("sigmas", torch.as_tensor(sigmas, dtype=torch.float32))
        self.sigma0 = float(sigma0)

    def forward(self, energy_fn, x, generator=None, create_graph=True):
        """Returns (loss, sigma_idx, per_sample_loss); the last two detached."""
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

        resid = x.detach() - x_tilde + (self.sigma0 ** 2) * grad
        per_sample = resid.pow(2).sum(dim=1) / sigma.squeeze(1).pow(2)
        return per_sample.mean(), idx.detach(), per_sample.detach()


class EnergyPrior(nn.Module):
    """Pretrained prior, frozen for task training (train_lifting --type dsm-prior).

    forward(y) chains bones -> [features] -> standardize -> E, so the task
    net's gradient is J^T grad E: each bone's gradient lands on its two
    endpoint joints with opposite signs, spring-force style. The wrapper
    carries the train-time standardizer statistics so task training cannot
    drift from the pretraining coordinates.

    With features="invariant" the energy is invariant to global scale and
    rotation, so by Euler's theorem its gradient never rescales the pose and
    exerts no net torque -- it can only change shape. Those are exactly the
    directions the 2026-08-26 sweep found useless, so removing them should
    raise the useful fraction of an already mostly-tangential force.
    """

    def __init__(self, parents, mean, std, hidden=512, depth=3, sigmas=None,
                 sigma0=0.1, features=None, lengths="symmetry",
                 joints_left=None, joints_right=None, noise_space="joint"):
        super(EnergyPrior, self).__init__()
        self.bones = BoneTransform(parents)
        self.noise_space = noise_space
        self.register_buffer("free", torch.as_tensor(
            [j for j in range(self.bones.num_joints) if j != self.bones.root],
            dtype=torch.long))
        self.features = None
        if features == "invariant":
            self.features = InvariantFeatures(
                parents, lengths=lengths, joints_left=joints_left,
                joints_right=joints_right,
            )
            in_dim = self.features.dim
        elif features in (None, "bone"):
            in_dim = self.bones.num_bones * 3
        else:
            raise ValueError(f"unknown features={features!r}")
        self.feature_mode = features or "bone"
        self.standardize = Standardizer(mean, std)
        self.net = EnergyNet(in_dim, hidden=hidden, depth=depth)
        self.register_buffer(
            "sigmas",
            torch.as_tensor(sigmas if sigmas is not None else [], dtype=torch.float32),
        )
        self.sigma0 = float(sigma0)
        self.meta = {}

    def corrupt_space(self, y):
        """Pose -> the coordinates pretraining added its noise to.

        Non-root joint positions or raw bone vectors (metres) for "invariant",
        standardized bone vectors for "bone". Gates and diagnostics work in
        this space so they probe the field where it was actually trained.
        """
        if self.features is None:
            return self.standardize(self.bones(y).flatten(1))
        if self.noise_space == "joint":
            y = y.view(y.shape[0], -1, 3)
            return (y - y[:, self.bones.root:self.bones.root + 1])[:, self.free].flatten(1)
        return self.bones(y).flatten(1)

    def _pose(self, x):
        """Corruption-space point -> full (B, J, 3) pose, root at the origin."""
        if self.noise_space == "joint" and self.features is not None:
            y = x.new_zeros(x.shape[0], self.bones.num_joints, 3)
            y[:, self.free] = x.view(x.shape[0], -1, 3)
            return y
        b = x if self.features is not None else self.standardize.inverse(x)
        return self.bones.compose(b.view(x.shape[0], -1, 3))

    def decode(self, x):
        """Inverse of corrupt_space, up to translation (root at the origin)."""
        return self._pose(x)

    def energy_at(self, x):
        """Energy of a point given in the corruption space."""
        if self.features is None:
            return self.net(x)
        return self.net(self.standardize(self.features(self.bones(self._pose(x)))))

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
            mean=ckpt["mean"],
            std=ckpt["std"],
            hidden=ckpt["hidden"],
            depth=ckpt["depth"],
            sigmas=ckpt["sigmas"],
            sigma0=ckpt["sigma0"],
            # Absent in checkpoints written before the invariant features
            # existed; those are all bone-vector priors.
            features=ckpt.get("features", "bone"),
            lengths=ckpt.get("lengths", "full"),
            joints_left=ckpt.get("joints_left"),
            joints_right=ckpt.get("joints_right"),
            noise_space=ckpt.get("noise_space", "bone"),
        )
        prior.net.load_state_dict(ckpt["net"])
        prior.meta = {k: ckpt[k] for k in ("train_args", "epoch", "val_dsm") if k in ckpt}
        return prior


def save_energy_checkpoint(path, net, parents, mean, std, sigmas, sigma0,
                           features="bone", lengths="full", joints_left=None,
                           joints_right=None, noise_space="bone",
                           train_args=None, epoch=None, val_dsm=None):
    """Single writer for the checkpoint format EnergyPrior.load expects."""
    torch.save(
        {
            "kind": "dsm_energy_prior",
            "features": features,
            "lengths": lengths,
            "noise_space": noise_space,
            "joints_left": None if joints_left is None else [int(j) for j in joints_left],
            "joints_right": None if joints_right is None else [int(j) for j in joints_right],
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
