# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
import h5py
import numpy as np
import copy

sys.path.append('../')

from utils.camera import normalize_screen_coordinates
from utils.mocap_dataset import MocapDataset
from utils.skeleton import Skeleton

# from common.skeleton import Skeleton
# from common.mocap_dataset import MocapDataset
# from common.camera import normalize_screen_coordinates, image_coordinates

mpi_inf_3dhp_skeleton = Skeleton(
    # parents=[1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 15],
    parents=[16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1],
    joints_left=[5, 6, 7, 11, 12, 13],
    joints_right=[2, 3, 4, 8, 9, 10],
    # parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15],
    # joints_left=[4, 5, 6, 11, 12, 13],
    # joints_right=[1, 2, 3, 14, 15, 16],
)


# mpi_inf_3dhp_skeleton = Skeleton(
#     parents=[
#         -1,
#         0,
#         1,
#         2,
#         3,
#         4,
#         0,
#         6,
#         7,
#         8,
#         9,
#         0,
#         11,
#         12,
#         13,
#         14,
#         12,
#         16,
#         17,
#         18,
#         19,
#         20,
#         19,
#         22,
#         12,
#         24,
#         25,
#         26,
#         27,
#         28,
#         27,
#         30,
#     ],
#     joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
#     joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
# )

subjects_train = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
subjects_test1 = ["TS1", "TS2", "TS3", "TS4"]
subjects_test2 = ["TS5", "TS6"]

mpi_inf_3dhp_cameras_intrinsic_params = [
    {
        "id": "cam_0",
        "center": [1024.704, 1051.394],
        "focal_length": [1497.693, 1497.103],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_1",
        "center": [1030.519, 1052.626],
        "focal_length": [1495.217, 1495.52],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_2",
        "center": [983.8873, 987.5902],
        "focal_length": [1495.587, 1497.828],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_3",
        "center": [1029.06, 1041.409],
        "focal_length": [1495.886, 1496.033],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": -110,  # Only used for visualization
    },
    {
        "id": "cam_4",
        "center": [987.6075, 1019.069],
        "focal_length": [1490.952, 1491.108],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_5",
        "center": [1012.331, 998.5009],
        "focal_length": [1500.414, 1499.971],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_6",
        "center": [999.7319, 1010.251],
        "focal_length": [1498.471, 1498.8],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_7",
        "center": [987.2716, 976.8773],
        "focal_length": [1498.831, 1499.674],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_8",
        "center": [1017.387, 1043.032],
        "focal_length": [1500.172, 1500.837],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_9",
        "center": [1010.423, 1037.096],
        "focal_length": [1501.554, 1501.9],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_10",
        "center": [1041.614, 997.0433],
        "focal_length": [1498.423, 1498.585],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_11",
        "center": [1009.802, 999.9984],
        "focal_length": [1495.779, 1493.703],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_12",
        "center": [1000.56, 1014.975],
        "focal_length": [1501.326, 1501.491],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "cam_13",
        "center": [1005.702, 1004.214],
        "focal_length": [1496.961, 1497.378],
        "radial_distortion": [0, 0, 0],
        "tangential_distortion": [0, 0],
        "res_w": 2048,
        "res_h": 2048,
        "azimuth": 70,  # Only used for visualization
    },
    {
        "id": "TS56",
        "center": [939.85754016, 560.140743168],
        "focal_length": [1683.98345952, 1672.59370772],
        "radial_distortion": [-0.276859611, 0.131125256, -0.049318332],
        "tangential_distortion": [-0.000360494, -0.001149441],
        "res_w": 1920,
        "res_h": 1080,
        "azimuth": 70,  # Only used for visualization
    },
]

mpi_inf_3dhp_cameras_extrinsic_params = {
    "Train": [
        {
            "orientation": [0.9910573, 0.0000989, 0.1322565, -0.017709],
            "translation": [-562.8666, 1398.138, 3852.623],
        },
        {
            "orientation": [0.8882246, -0.0698901, 0.4388433, -0.1165721],
            "translation": [-1429.856, 738.1779, 4897.966],
        },
        {
            "orientation": [0.5651277, -0.0301201, 0.824319, -0.0148915],
            "translation": [57.25702, 1307.287, 2799.822],
        },
        {
            "orientation": [0.6670245, -0.1827152, 0.7089925, -0.1379241],
            "translation": [-284.8168, 807.9184, 3177.16],
        },
        {
            "orientation": [0.8273998, 0.0263385, 0.5589656, -0.0476783],
            "translation": [-1563.911, 801.9608, 3517.316],
        },
        {
            "orientation": [-0.568842, 0.0159665, 0.8220693, -0.0191314],
            "translation": [358.4134, 994.5658, 3439.832],
        },
        {
            "orientation": [0.2030824, -0.2818073, 0.9370704, -0.0352313],
            "translation": [569.4388, 528.871, 3687.369],
        },
        {
            "orientation": [0.00086, 0.0123344, 0.9998223, -0.0142292],
            "translation": [1378.866, 1270.781, 2631.567],
        },
        {
            "orientation": [0.7053718, 0.095632, -0.7004048, -0.0523286],
            "translation": [221.3543, 659.87, 3644.688],
        },
        {
            "orientation": [0.6914033, 0.2036966, -0.6615312, -0.2069921],
            "translation": [388.6217, 137.5452, 4216.635],
        },
        {
            "orientation": [-0.2266321, -0.2540748, 0.9401911, -0.0111636],
            "translation": [1167.962, 617.6362, 4472.351],
        },
        {
            "orientation": [-0.4536946, -0.2035304, -0.0072578, 0.8675736],
            "translation": [134.8272, 251.5094, 4570.244],
        },
        {
            "orientation": [-0.0778876, 0.8469901, -0.4230185, 0.3124046],
            "translation": [412.4695, 532.7588, 4887.095],
        },
        {
            "orientation": [0.098712, 0.8023286, -0.5397436, -0.2349501],
            "translation": [867.1278, 827.4572, 3985.159],
        },
    ],
    "chestHeight": [
        {
            "orientation": [0.7053718, 0.095632, -0.7004048, -0.0523286],
            "translation": [221.3543, 659.87, 3644.688],
        },
    ],
}


class MpiInf3dhpDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        super().__init__(fps=25, skeleton=copy.deepcopy(mpi_inf_3dhp_skeleton))

        self._cameras = {}

        for subject in subjects_train:
            self._cameras[subject] = copy.deepcopy(
                mpi_inf_3dhp_cameras_extrinsic_params["Train"]
            )

        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(mpi_inf_3dhp_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ["id", "res_w", "res_h"]:
                        cam[k] = np.array(v, dtype="float32")

                # Normalize camera frame
                cam["center"] = normalize_screen_coordinates(
                    cam["center"], w=cam["res_w"], h=cam["res_h"]
                ).astype("float32")
                cam["focal_length"] = cam["focal_length"] / cam["res_w"] * 2
                if "translation" in cam:
                    cam["translation"] = cam["translation"] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam["intrinsic"] = np.concatenate(
                    (
                        cam["focal_length"],
                        cam["center"],
                        cam["radial_distortion"],
                        cam["tangential_distortion"],
                    )
                )

        for subject in subjects_test1:
            self._cameras[subject] = copy.deepcopy(
                mpi_inf_3dhp_cameras_extrinsic_params["chestHeight"]
            )
            cam = self._cameras[subject][0]
            cam.update(mpi_inf_3dhp_cameras_intrinsic_params[8])
            for k, v in cam.items():
                if k not in ["id", "res_w", "res_h"]:
                    cam[k] = np.array(v, dtype="float32")

            # Normalize camera frame
            cam["center"] = normalize_screen_coordinates(
                cam["center"], w=cam["res_w"], h=cam["res_h"]
            ).astype("float32")
            cam["focal_length"] = cam["focal_length"] / cam["res_w"] * 2
            if "translation" in cam:
                cam["translation"] = cam["translation"] / 1000  # mm to meters

            # Add intrinsic parameters vector
            cam["intrinsic"] = np.concatenate(
                (
                    cam["focal_length"],
                    cam["center"],
                    cam["radial_distortion"],
                    cam["tangential_distortion"],
                )
            )

        for subject in subjects_test2:
            self._cameras[subject] = copy.deepcopy(
                mpi_inf_3dhp_cameras_extrinsic_params["chestHeight"]
            )
            cam = self._cameras[subject][0]
            cam.update(mpi_inf_3dhp_cameras_intrinsic_params[14])
            for k, v in cam.items():
                if k not in ["id", "res_w", "res_h"]:
                    cam[k] = np.array(v, dtype="float32")

            # Normalize camera frame
            cam["center"] = normalize_screen_coordinates(
                cam["center"], w=cam["res_w"], h=cam["res_h"]
            ).astype("float32")
            cam["focal_length"] = cam["focal_length"] / cam["res_w"] * 2
            if "translation" in cam:
                cam["translation"] = cam["translation"] / 1000  # mm to meters

            # Add intrinsic parameters vector
            cam["intrinsic"] = np.concatenate(
                (
                    cam["focal_length"],
                    cam["center"],
                    cam["radial_distortion"],
                    cam["tangential_distortion"],
                )
            )

        # Load serialized dataset
        data = np.load(path, allow_pickle=True)["positions_3d"].item()

        self._data = {}

        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    "positions": positions,
                    "cameras": self._cameras[subject],
                }

        # if remove_static_joints:
        #     # Bring the skeleton to 17 joints instead of the original 32
        #     self._skeleton.remove_joints(
        #         [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
        #     )

        #     # Rewire shoulders to the correct parents
        #     self._skeleton._parents[11] = 8
        #     self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return True


output_filename = "data_3d_mpi_inf_3dhp"
output_filename_2d = "data_2d_mpi_inf_3dhp_gt"
output_filename_2d2 = "data_2d_mpi_inf_3dhp_computed_gt"
subjects_train = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
subjects_test = ["TS1", "TS2", "TS3", "TS4", "TS5", "TS6"]
joint_idx_train_matlab = [
    8,
    6,
    15,
    16,
    17,
    10,
    11,
    12,
    24,
    25,
    26,
    19,
    20,
    21,
    5,
    4,
    7,
]  # notice: it is in matlab index
joint_idx_train = [i - 1 for i in joint_idx_train_matlab]

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "data":
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(
        description="MPI_INF_3DHP dataset downloader/converter"
    )

    # Convert dataset from original source, using files converted to .mat (the Human3.6M dataset path must be specified manually)
    # This option requires MATLAB to convert files using the provided script
    parser.add_argument(
        "--from-source",
        default="",
        type=str,
        metavar="PATH",
        help="convert original dataset",
    )

    args = parser.parse_args()

    if os.path.exists(output_filename + ".npz"):
        print("The dataset already exists at", output_filename + ".npz")
        exit(0)

    if args.from_source:
        print("Converting original MPI_INF_3DHP dataset from", args.from_source)
        output = {}
        output_2d_poses = {}
        from scipy.io import loadmat

        for subject in subjects_train:
            output[subject] = {}
            output_2d_poses[subject] = {}
            file_1 = args.from_source + "/" + subject + "/Seq1/annot.mat"
            file_2 = args.from_source + "/" + subject + "/Seq2/annot.mat"
            hf = loadmat(file_1)
            positions_3d_temp = []
            positions_2d_temp = []

            for index in range(14):
                positions = hf["annot3"][index, 0].reshape(-1, 28, 3)
                positions /= 1000  # Meters instead of millimeters
                positions_17 = positions[:, joint_idx_train, :]
                positions_17[:, 1:] -= positions_17[
                    :, :1
                ]  # Remove global offset, but keep trajectory in first position
                positions_3d_temp.append(positions_17.astype("float32"))
                positions_2d = hf["annot2"][index, 0].reshape(-1, 28, 2)
                positions_2d_temp.append(
                    positions_2d[:, joint_idx_train, :].astype("float32")
                )

            output[subject]["Seq1"] = positions_3d_temp
            output_2d_poses[subject]["Seq1"] = positions_2d_temp

            positions_3d_temp = []
            positions_2d_temp = []
            hf = loadmat(file_2)
            for index in range(14):
                positions = hf["annot3"][index, 0].reshape(-1, 28, 3)
                positions /= 1000  # Meters instead of millimeters
                positions_17 = positions[:, joint_idx_train, :]
                positions_17[:, 1:] -= positions_17[
                    :, :1
                ]  # Remove global offset, but keep trajectory in first position
                positions_3d_temp.append(positions_17.astype("float32"))
                positions_2d = hf["annot2"][index, 0].reshape(-1, 28, 2)
                positions_2d_temp.append(
                    positions_2d[:, joint_idx_train, :].astype("float32")
                )
            output[subject]["Seq2"] = positions_3d_temp
            output_2d_poses[subject]["Seq2"] = positions_2d_temp

        for subject in subjects_test:
            output[subject] = {}
            output_2d_poses[subject] = {}
            file_1 = (
                args.from_source
                + "/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/"
                + subject
                + "/annot_data.mat"
            )
            hf = {}
            f = h5py.File(file_1)
            for k, v in f.items():
                hf[k] = np.array(v)
            positions = hf["annot3"].reshape(-1, 17, 3)
            positions /= 1000  # Meters instead of millimeters
            positions_17 = positions
            positions_17[:, 1:] -= positions_17[
                :, :1
            ]  # Remove global offset, but keep trajectory in first position
            output[subject]["Test"] = [positions_17.astype("float32")]
            positions_2d = hf["annot2"].reshape(-1, 17, 2)
            output_2d_poses[subject]["Test"] = [positions_2d.astype("float32")]

        print("Saving...")
        np.savez_compressed(output_filename, positions_3d=output)
        print("")
        print("Getting 2D poses...")
        dataset = MpiInf3dhpDataset(output_filename + ".npz")
        metadata = {
            "num_joints": dataset.skeleton().num_joints(),
            "keypoints_symmetry": [
                dataset.skeleton().joints_left(),
                dataset.skeleton().joints_right(),
            ],
        }
        print("Saving...")
        np.savez_compressed(
            output_filename_2d, positions_2d=output_2d_poses, metadata=metadata
        )

        print("Done.")
    else:
        print("Please specify the dataset source")
        exit(0)
"""
    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = MpiInf3dhpDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            positions_2d = []
            for i,cam in enumerate(anim['cameras']):
                pos_3d = anim['positions'][i]
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d
            
    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d2, positions_2d=output_2d_poses, metadata=metadata)
"""
