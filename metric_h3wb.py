import argparse
import torch
from evaluate import prepare_data_h3wb_val
from utils.device import get_device
from tqdm.auto import tqdm
import numpy as np
from utils.seed import set_seed

lefts_body = [
    (12, 14),
    (14, 16),
    (6, 8),
    (8, 10),
]
rights_body = [
    (13, 15),
    (15, 17),
    (7, 9),
    (9, 11),
]
lefts_foot = [
    (18, 19),
    (19, 20),
    (18, 20),
]
rights_foot = [
    (21, 22),
    (22, 23),
    (21, 23),
]
lefts_hand = [
    (92, 93),
    (93, 94),
    (94, 95),
    (95, 96),
    (97, 98),
    (98, 99),
    (99, 100),
    (101,102),
    (102,103),
    (103,104),
    (105,106),
    (106,107),
    (107,108),
    (109,110),
    (110,111),
    (111,112),
]
rights_hand = [
    (113,114),
    (114,115),
    (115,116),
    (116,117),
    (118,119),
    (119,120),
    (120,121),
    (122,123),
    (123,124),
    (124,125),
    (126,127),
    (127,128),
    (128,129),
    (130,131),
    (131,132),
    (132,133),
]
lefts_face = [
    (32, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    (36, 37),
    (37, 38),
    (38, 39),
    (39, 40),
    (46, 47),
    (47, 48),
    (48, 49),
    (49, 50),
    (66, 67),
    (67, 68),
    (68, 69),
    (69, 70),
    (70, 71),
    (71, 66),
    (57, 58),
    (58, 59),
    (75, 76),
    (76, 77),
    (77, 78),
    (81, 80),
    (80, 79),
    (79, 78),
    (86, 87),
    (87, 88),
]
rights_face = [
    (32, 31),
    (31, 30),
    (30, 29),
    (29, 28),
    (28, 27),
    (27, 26),
    (26, 25),
    (25, 24),
    (45, 44),
    (44, 43),
    (43, 42),
    (42, 41),
    (63, 62),
    (62, 61),
    (61, 60),
    (60, 65),
    (65, 64),
    (64, 63),
    (57, 56),
    (56, 55),
    (75, 74),
    (74, 73),
    (73, 72),
    (81, 82),
    (82, 83),
    (83, 84),
    (86, 85),
    (85, 84),
]

def check_length_symmetry(keypoints, option="all"):
    keypoints = keypoints.reshape(-1, 3)
    if option == "all":
        lefts = lefts_body + lefts_foot + lefts_hand
        rights = rights_body + rights_foot + rights_hand
    elif option == "body":
        lefts = lefts_body
        rights = rights_body
    elif option == "foot":
        lefts = lefts_foot
        rights = rights_foot
    elif option == "hand":
        lefts = lefts_hand
        rights = rights_hand
    elif option == "face":
        lefts = lefts_face
        rights = rights_face

    for i in range(len(lefts)):
        left_length = np.linalg.norm(
            keypoints[lefts[i][0] - 1] - keypoints[lefts[i][1] - 1]
        )
        right_length = np.linalg.norm(
            keypoints[rights[i][0] - 1] - keypoints[rights[i][1] - 1]
        )
        violations = []
        violations.append(
            np.abs(left_length - right_length) / (left_length + right_length)
        )

    return violations


def check_joint_angles(keypoints, targets, option="all"):
    keypoints = keypoints.reshape(-1, 3)
    targets = targets.reshape(-1, 3)
    angles = []
    angles_label = []
    diffs = []
    if option == "all":
        joints = lefts_body + rights_body + lefts_foot + rights_foot + lefts_hand + rights_hand
    elif option == "body":
        joints = lefts_body + rights_body
    elif option == "foot":
        joints = lefts_foot + rights_foot
    elif option == "hand":
        joints = lefts_hand + rights_hand
    elif option == "face":
        joints = lefts_face + rights_face
        
    adjacent_triplets_h3wb = []
    for joint1 in joints:
        for joint2 in joints:
            if joint1[1] == joint2[0]:
                adjacent_triplets_h3wb.append((joint1[0], joint1[1], joint2[1]))
        
    for triplet in adjacent_triplets_h3wb:
        vec1 = keypoints[triplet[0]-1] - keypoints[triplet[1]-1]
        vec2 = keypoints[triplet[2] -1] - keypoints[triplet[1]-1]
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
        
        vec1_gold = targets[triplet[0]-1] - targets[triplet[1]-1]
        vec2_gold = targets[triplet[2] -1] - targets[triplet[1]-1]
        cos_angle_gold = np.dot(vec1_gold, vec2_gold) / (
            np.linalg.norm(vec1_gold) * np.linalg.norm(vec2_gold)
        )
        cos_angle_gold = np.clip(cos_angle_gold, -1.0, 1.0)
        angle_gold = np.arccos(cos_angle_gold)
        angles_label.append(angle_gold)
        diffs.append(np.abs(angle - angle_gold))
        # diffs.append(np.abs(angle - angle_gold) / angle_gold)

    return diffs


def check_joint_distances(keypoints, targets, option="all"):
    keypoints = keypoints.reshape(-1, 3)
    targets = targets.reshape(-1, 3)
    distances = []
    distances_label = []
    diffs = []

    if option == "all":
        joints = lefts_body + rights_body + lefts_foot + rights_foot + lefts_hand + rights_hand
    elif option == "body":
        joints = lefts_body + rights_body
    elif option == "foot":
        joints = lefts_foot + rights_foot
    elif option == "hand":
        joints = lefts_hand + rights_hand
    elif option == "face":
        joints = lefts_face + rights_face
        
    adjacent_triplets_h3wb = []
    for joint1 in joints:
        for joint2 in joints:
            if joint1[1] == joint2[0]:
                adjacent_triplets_h3wb.append((joint1[0], joint1[1], joint2[1]))

    for triplet in adjacent_triplets_h3wb:
        dist1 = np.linalg.norm(keypoints[triplet[1]-1] - keypoints[triplet[0]-1])
        dist2 = np.linalg.norm(keypoints[triplet[2]-1] - keypoints[triplet[1]-1])
        distances += [dist1, dist2]
        dist1_gold = np.linalg.norm(targets[triplet[1]-1] - targets[triplet[0]-1])
        dist2_gold = np.linalg.norm(targets[triplet[2]-1] - targets[triplet[1]-1])
        distances_label += [dist1_gold, dist2_gold]
        diffs += [np.abs(dist1 - dist1_gold) / dist1_gold, np.abs(dist2 - dist2_gold) / dist2_gold]

    # for i in range(len(distances)):
    #     diffs.append(np.abs(distances[i] - distances_label[i]) / distances_label[i])

    return diffs


def main(args):
    set_seed(args.seed)
    device = get_device()
    # device = torch.device("cpu")
    args.eval_batch_size = args.batch_size
    eval_dataloader, _dataset = prepare_data_h3wb_val(
        args, "data/h3wb_train.npz", "data/task1_test_3d.npz"
    )

    try:
        net.load_state_dict(torch.load(args.checkpoint), map_location=device)
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
            syms += check_length_symmetry(outputs[i].detach().cpu().numpy(), option=args.option)
            syms_gold += check_length_symmetry(targets_3d[i].detach().cpu().numpy(), option=args.option)
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
    print(len(syms), len(syms_gold), len(diffs_angle), len(diffs_dist))

    sym_count = np.count_nonzero(syms > args.threshold) / len(syms) * 100
    sym_count_gold = np.count_nonzero(syms_gold > args.threshold) / len(syms_gold) * 100
    angle_count = np.count_nonzero(diffs_angle > args.threshold) / len(diffs_angle) * 100
    dist_count = np.count_nonzero(diffs_dist > args.threshold) / len(diffs_dist) * 100
    
    print(f"==> type: {args.option} checkpoint: {args.checkpoint}")
    print(f"Symmetry golds: {sym_count_gold:.2f}%, {np.mean(syms_gold) * 100:.2f}%")
    print(f"Symmetry diffs: {sym_count:.2f}%, {np.mean(syms) * 100:.2f}%")
    print(f"Angle    diffs: {angle_count:.2f}%, {np.mean(diffs_angle) * 180/np.pi:.2f}° ")
    print(f"Distance diffs: {dist_count:.2f}%, {np.mean(diffs_dist) * 100:.2f}%\n")

    # print(f"Symmetry diffs: {sym_count:.3E}, {np.mean(syms):.3E} (gold {np.mean(syms_gold):.3E})")
    # print(f"Angle    diffs: {angle_count:.3E}, {np.mean(diffs_angle):.3E} (std  {np.std(diffs_angle):.3E})")
    # print(f"Distance diffs: {dist_count:.3E}, {np.mean(diffs_dist):.3E} (std  {np.std(diffs_dist):.3E})\n")
    return (sym_count, np.mean(syms), angle_count, np.mean(diffs_angle), dist_count, np.mean(diffs_dist))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="dynamic")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_data", type=int, default=1)
    parser.add_argument("--keypoints", type=str, default="gt")
    parser.add_argument("--option", type=str, default="all")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
