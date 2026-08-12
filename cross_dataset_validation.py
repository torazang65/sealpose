

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import PoseBuffer
from eval import evaluate
from utils.device import get_device


def main():
    mpi3d_npz = np.load("data_extra/test_set/test_3dhp.npz")
    tmp = mpi3d_npz
    mpi3d_loader = DataLoader(
        PoseBuffer([tmp["pose3d"]], [tmp["pose2d"]]),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    device = get_device()
    
    model_pos = torch.load(args.checkpoint).to(device)
    
    error_3dhp_p1, error_3dhp_p2 = evaluate(mpi3d_loader, model_pos, device)
    print('3DHP: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p1))
    print('3DHP: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p2))

    error_3dhp_p1, error_3dhp_p2 = evaluate(mpi3d_loader, model_pos, device, flipaug='_flip')
    print('3DHP: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p1))
    print('3DHP: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint/model_pos.pt")
    args = parser.parse_args()
    print(args)

    main(args)