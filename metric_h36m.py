import argparse
import torch
from evaluate import prepare_data_h36m_val
from models.linear_model import LinearModel
from utils.device import get_device
from tqdm.auto import tqdm
import numpy as np
from utils.seed import set_seed


def check_length_symmetry(keypoints, option="all"):
    keypoints = keypoints.reshape(-1, 3)
    if option == "all":
        lefts = [
            (10, 11),
            (11, 12),
            (8, 10),
            (0, 4),
            (4, 5),
            (5, 6),
        ]
        rights = [
            (13, 14),
            (14, 15),
            (8, 13),
            (0, 1),
            (1, 2),
            (2, 3),
        ]

        for i in range(len(lefts)):
            left_length = np.linalg.norm(
                keypoints[lefts[i][0]] - keypoints[lefts[i][1]]
            )
            right_length = np.linalg.norm(
                keypoints[rights[i][0]] - keypoints[rights[i][1]]
            )
            violations = []
            violations.append(
                np.abs(left_length - right_length) / (left_length + right_length)
            )
    elif option == "limb":
        left_arm = (10, 11, 12)
        right_arm = (13, 14, 15)
        left_leg = (4, 5, 6)
        right_leg = (1, 2, 3)

        left_arm_length = np.linalg.norm(
            keypoints[left_arm[0]] - keypoints[left_arm[1]]
        ) + np.linalg.norm(keypoints[left_arm[1]] - keypoints[left_arm[2]])
        right_arm_length = np.linalg.norm(
            keypoints[right_arm[0]] - keypoints[right_arm[1]]
        ) + np.linalg.norm(keypoints[right_arm[1]] - keypoints[right_arm[2]])
        left_leg_length = np.linalg.norm(
            keypoints[left_leg[0]] - keypoints[left_leg[1]]
        ) + np.linalg.norm(keypoints[left_leg[1]] - keypoints[left_leg[2]])
        right_leg_length = np.linalg.norm(
            keypoints[right_leg[0]] - keypoints[right_leg[1]]
        ) + np.linalg.norm(keypoints[right_leg[1]] - keypoints[right_leg[2]])
        violations = []
        violations.append(
            np.abs(left_arm_length - right_arm_length)
            / (left_arm_length + right_arm_length)
        )
        violations.append(
            np.abs(left_leg_length - right_leg_length)
            / (left_leg_length + right_leg_length)
        )

    return violations


adjacent_triplets_h36m = [
    (1, 0, 4),
    (0, 1, 2),
    (1, 2, 3),
    (0, 4, 5),
    (4, 5, 6),
    (1, 0, 7),
    (4, 0, 7),
    
    (0, 7, 8),
    (7, 8, 9),
    (7, 8, 10),
    (7, 8, 13),
    (9, 8, 10),
    (9, 8, 13),
    
    (10, 8, 13),
    (8, 13, 14),
    (13, 14, 15),
    (8, 10, 11),
    (10, 11, 12),
]


def check_joint_angles(keypoints, targets):
    keypoints = keypoints.reshape(-1, 3)
    targets = targets.reshape(-1, 3)
    angles = []
    angles_label = []
    diffs = []
    for triplet in adjacent_triplets_h36m:
        vec1 = keypoints[triplet[0]] - keypoints[triplet[1]]
        vec2 = keypoints[triplet[2]] - keypoints[triplet[1]]
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
        
        vec1_gold = targets[triplet[0]] - targets[triplet[1]]
        vec2_gold = targets[triplet[2]] - targets[triplet[1]]
        cos_angle_gold = np.dot(vec1_gold, vec2_gold) / (
            np.linalg.norm(vec1_gold) * np.linalg.norm(vec2_gold)
        )
        cos_angle_gold = np.clip(cos_angle_gold, -1.0, 1.0)
        angle_gold = np.arccos(cos_angle_gold)
        angles_label.append(angle_gold)
        # print(
        #     f"angle: {angle * 180 / np.pi:.2f}, angle_gold: {angle_gold * 180 / np.pi:.2f}, part: {triplet}"
        # )
        diffs.append(np.abs(angle - angle_gold))

    return diffs


def check_joint_distances(keypoints, targets):
    keypoints = keypoints.reshape(-1, 3)
    targets = targets.reshape(-1, 3)
    distances = []
    distances_label = []
    diffs = []
    for triplet in adjacent_triplets_h36m:
        dist1 = np.linalg.norm(keypoints[triplet[1]] - keypoints[triplet[0]])
        dist2 = np.linalg.norm(keypoints[triplet[2]] - keypoints[triplet[1]])
        distances += [dist1, dist2]
        dist1_gold = np.linalg.norm(targets[triplet[1]] - targets[triplet[0]])
        dist2_gold = np.linalg.norm(targets[triplet[2]] - targets[triplet[1]])
        distances_label += [dist1_gold, dist2_gold]
        diffs += [np.abs(dist1 - dist1_gold) / dist1_gold, np.abs(dist2 - dist2_gold) / dist2_gold]

    return diffs


def main(args):
    set_seed(args.seed)
    # device = get_device()
    device = torch.device("cpu")
    args.eval_batch_size = args.batch_size
    eval_dataloader, _dataset = prepare_data_h36m_val(args, "data/data_3d_h36m.npz")


    try:
        net = LinearModel(
            16 * 2,
            (16 - 1) * 3,
            linear_size=1024,
            num_stage=2,
            p_dropout=0.5,
            num_joints=16,
            batch_norm=1,
        ).to(device)
        # print(torch.load(args.checkpoint, map_location=device))
        net.load_state_dict(torch.load(args.checkpoint, map_location=device)['state_dict'])
    except:
        net = torch.load(args.checkpoint, map_location=device)

    net.eval()
    count = 0
    syms = []
    syms_gold = []
    diffs_angle = []
    diffs_dist = []

    for data in tqdm(eval_dataloader):
        targets_3d, inputs_2d = data[0].to(device), data[1].to(device)
        batch_size = inputs_2d.shape[0]
        outputs = net(inputs_2d)
        outputs = outputs.reshape(batch_size, -1)
        targets_3d = targets_3d.reshape(batch_size, -1)
        for i in range(batch_size):
            syms += check_length_symmetry(outputs[i].detach().cpu().numpy() ,option=args.option)
            # syms_gold += check_length_symmetry(targets_3d[i].detach().cpu().numpy())
            diffs_angle += check_joint_angles(
                outputs[i].detach().cpu().numpy(), targets_3d[i].detach().cpu().numpy(), option=args.option
            )
            diffs_dist += check_joint_distances(
                outputs[i].detach().cpu().numpy(), targets_3d[i].detach().cpu().numpy(), option=args.option
            )

        count += 1
        if count == args.num_data:
            break
    syms, syms_gold, diffs_angle, diffs_dist = (
        np.array(syms),
        np.array(syms_gold),
        np.array(diffs_angle),
        np.array(diffs_dist),
    )

    sym_count = np.count_nonzero(syms > 0.01) / len(syms) * 100
    angle_count = np.count_nonzero(diffs_angle > 0.1) / len(diffs_angle) * 100
    dist_count = np.count_nonzero(diffs_dist > 0.1) / len(diffs_dist) * 100
    
    print("==> checkpoint: ", args.checkpoint)
    print(f"Symmetry diffs: {sym_count:.2f}%, {np.mean(syms) * 100:.2f}%")
    print(f"Angle    diffs: {angle_count:.2f}%, {np.mean(diffs_angle) * 180/np.pi:.2f}° ")
    print(f"Distance diffs: {dist_count:.2f}%, {np.mean(diffs_dist) * 100:.2f}%\n")

    # print(f"Symmetry diffs: {sym_count:.3E}, {np.mean(syms):.3E} (gold {np.mean(syms_gold):.3E})")
    # print(f"Angle    diffs: {angle_count:.3E}, {np.mean(diffs_angle):.3E} (std  {np.std(diffs_angle):.3E})")
    # print(f"Distance diffs: {dist_count:.3E}, {np.mean(diffs_dist):.3E} (std  {np.std(diffs_dist):.3E})\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="dynamic")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_data", type=int, default=1)
    parser.add_argument("--option", type=str, default=None)
    parser.add_argument("--keypoints", type=str, default="gt")
    args = parser.parse_args()
    main(args)
