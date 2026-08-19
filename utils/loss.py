from __future__ import absolute_import, division

import numpy as np
import torch
import torch.nn as nn

def test_score(predict_list, target_list, verbose=True):
    count = [0, 0, 0, 0, 0, 0]
    diff = predict_list - target_list
    diff = diff - (diff[:, 11:12, :] + diff[:, 12:13, :]) / 2  # pelvis align
    diff1 = (diff - diff[:, 0:1, :])[:, 23:91, :]  # nose align face
    diff21 = (diff - diff[:, 91:92, :])[:, 91:112, :]  # wrist aligned left hand
    diff22 = (diff - diff[:, 112:113, :])[:, 112:, :]  # wrist aligned right hand

    diff = torch.sqrt(torch.sum(torch.square(diff), dim=-1))

    count[0] = torch.mean(diff).item() * 1000
    count[1] = torch.mean(diff[:, :23]).item() * 1000
    count[2] = torch.mean(diff[:, 23:91]).item() * 1000
    count[3] = torch.mean(diff[:, 91:]).item() * 1000
    count[4] = torch.mean(torch.sqrt(torch.sum(torch.square(diff1), dim=-1))).item() * 1000
    count[5] = (
        torch.mean(torch.sqrt(torch.sum(torch.square(diff21), dim=-1))).item()
        + torch.mean(torch.sqrt(torch.sum(torch.square(diff22), dim=-1))).item()
    ) * 1000

    if verbose:
        print("Pelvis aligned MPJPE is " + str(count[0]) + " mm")
        print("Pelvis aligned MPJPE on body is " + str(count[1]) + " mm")
        print("Pelvis aligned MPJPE on face is " + str(count[2]) + " mm")
        print("Nose aligned MPJPE on face is " + str(count[4]) + " mm")
        print("Pelvis aligned MPJPE on hands is " + str(count[3]) + " mm")
        print("Wrist aligned MPJPE on hands is " + str(count[5] / 2) + " mm")

    return (count[0], count[1], count[2], count[3], count[4], count[5] / 2)


class BoneDirectionLoss(nn.Module):
    """Length-weighted squared error of each bone's direction.

    For bone b = (joint j, its parent), with GT length l_b and unit vectors
    u_hat_b / u_b, the direction-only displacement is

        d_b = l_b * (u_hat_b - u_b),      ||d_b|| = 2 * l_b * sin(theta_b / 2)

    i.e. exactly the positional error that survives when the bone length is
    correct but the direction is not. l_b comes from the ground truth, so it
    acts as a constant weight: this term has zero gradient with respect to
    predicted bone length and only supervises direction.

    Bones are weighted by c_b = 1 + (number of joints hanging off b), which is
    the weight position-MSE already gives a bone's direction implicitly -- a
    direction error there displaces every descendant. Keeping c_b therefore
    preserves the spatial weighting of the MSE term and only raises the
    direction half relative to the length half, which is what the headroom
    measurement in report/check_bodyness.py calls for (direction -21.98 mm vs
    length -9.09 mm on gcn-base-d095).

    Normalised per coordinate, so it is directly comparable to
    nn.MSELoss(reduction="mean") and a weight of 1.0 puts the two on an equal
    footing.

    Args:
        parents:  parent index per joint, -1 for the root.
        exclude:  joints to drop. For 3DHP pass [0] -- ground truth puts joint 0
                  1.2-6.3 m from the head, so it is not a body joint, and an
                  l^2 weight would hand it ~42x the weight of a femur.
    """

    def __init__(self, parents, exclude=()):
        super(BoneDirectionLoss, self).__init__()
        parents = list(parents)
        exclude = set(exclude)
        bones = [j for j, p in enumerate(parents)
                 if p != -1 and j not in exclude and p not in exclude]

        def num_descendants(j):
            n, stack = 0, [j]
            while stack:
                cur = stack.pop()
                kids = [k for k in bones if parents[k] == cur]
                n += len(kids)
                stack += kids
            return n

        self.register_buffer("child", torch.as_tensor(bones, dtype=torch.long))
        self.register_buffer("parent", torch.as_tensor([parents[j] for j in bones],
                                                       dtype=torch.long))
        self.register_buffer("weight", torch.as_tensor(
            [1.0 + num_descendants(j) for j in bones], dtype=torch.float32))
        self.eps = 1e-6

    def forward(self, predicted, target):
        predicted = predicted.view(predicted.shape[0], -1, 3)
        target = target.view(target.shape[0], -1, 3)

        vp = predicted[:, self.child] - predicted[:, self.parent]
        vt = target[:, self.child] - target[:, self.parent]

        lt = vt.norm(dim=-1, keepdim=True)                       # GT length: constant
        up = vp / vp.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        ut = vt / lt.clamp_min(self.eps)

        d2 = ((up - ut) * lt).pow(2).sum(-1)                     # (B, n_bones)
        return (d2 * self.weight).sum(-1).mean() / (3.0 * self.weight.sum())


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def mpjpe_byjoint(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=0)


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Reimplementation of MATLAB's `procrustes` function to Numpy,
    refer to https://codeday.me/bug/20180920/256259.html

    Args
        X: target pose
        Y: input pose
        scaling: if False, the scaling component of the transformation is forced to 1
        reflection: if 'best' (default), the transformation solution may or may not
                    include a reflection component, depending on which fits the data
                    best. setting reflection to True or False forces a solution with
                    reflection or no reflection respectively.
    Return
        d: the residual sum of squared errors, normalized according to a
           measure of the scale of X, ((X - X.mean(0))**2).sum()
        Z: the matrix of transformed Y-values
        tform: a dict specifying the rotation, translation and scaling that maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def p_mpjpe(predicted, target):
    # return torch.from_numpy(np.array(0.05))  # kh0826
    # assert False, 'skip this mean to save time'
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))


def compute_PCK(gts, preds, scales=1000, eval_joints=None, threshold=150):
    PCK_THRESHOLD = threshold
    sample_num = len(gts)
    total = 0
    true_positive = 0
    if eval_joints is None:
        eval_joints = list(range(gts.shape[1]))

    for n in range(sample_num):
        gt = gts[n]
        pred = preds[n]
        # scale = scales[n]
        scale = 1000
        per_joint_error = np.take(np.sqrt(np.sum(np.power(pred - gt, 2), 1)) * scale, eval_joints, axis=0)
        true_positive += (per_joint_error < PCK_THRESHOLD).sum()
        total += per_joint_error.size

    pck = float(true_positive / total) * 100
    return pck


def compute_AUC(gts, preds, scales=1000, eval_joints=None):
    # This range of thresholds mimics 'mpii_compute_3d_pck.m', which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = np.linspace(0, 150, 31)
    pck_list = []
    for threshold in thresholds:
        pck_list.append(compute_PCK(gts, preds, scales, eval_joints, threshold))

    auc = np.mean(pck_list)

    return auc

def diff_range_loss(a, b, std):
    diff = (a - b) ** 2
    weight = torch.where(diff > std ** 2, torch.ones_like(a), torch.zeros_like(a))
    diff_weighted = diff * weight
    return diff_weighted.mean()


def rectifiedL2loss(gamma, threshold):  # threshold = b
    diff = (gamma - 0) ** 2
    weight = torch.where(diff > threshold ** 2, torch.ones_like(gamma), torch.zeros_like(gamma))
    diff_weighted = diff * weight
    return diff_weighted.mean()


if __name__ == '__main__':
    a = torch.randn(2, 15, 3, requires_grad=True)
    b = torch.randn(2, 15, 3)
    # cir = Weighted_mse_loss()
    # loss = cir(a, b)
    # # loss = weighted_mse_loss(a, b)
    # loss.backward()
    print('done')