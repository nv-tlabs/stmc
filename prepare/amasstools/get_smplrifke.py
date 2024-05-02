import os
import argparse
import numpy as np
import torch

from loop_amass import loop_amams
from smplrifke_feats import smpldata_to_smplrifkefeats, smplrifkefeats_to_smpldata


def get_features(folder, base_name, joints_name, new_name, force_redo=False):
    print("Get smplrifke features")
    print("The processed motions will be stored in this folder:")

    base_folder = os.path.join(folder, base_name)
    joints_folder = os.path.join(folder, joints_name)
    new_folder = os.path.join(folder, new_name)

    if not os.path.exists(base_folder):
        print(f"{base_folder} folder does not exist")
        print("Run fix_fps.py")
        exit()

    if not os.path.exists(joints_folder):
        print(f"{joints_folder} folder does not exist")
        print("Run extract_joints.py")
        exit()

    print(new_folder)

    iterator = loop_amams(
        base_folder, new_folder, ext=".npz", newext=".npy", force_redo=force_redo
    )

    for motion_path, new_motion_path in iterator:
        # smpl
        smpl_data = np.load(motion_path)
        # joints
        joint_path = motion_path.replace(base_name, joints_name).replace(".npz", ".npy")
        joints = np.load(joint_path)

        # assert no hands
        assert smpl_data["poses"].shape[-1] == 66

        smpl_data = {
            "poses": torch.from_numpy(smpl_data["poses"]).to(torch.double),
            "trans": torch.from_numpy(smpl_data["trans"]).to(torch.double),
            "joints": torch.from_numpy(joints).to(torch.double),
        }

        # apply transformation
        try:
            features = smpldata_to_smplrifkefeats(smpl_data)
        except IndexError:
            # The sequence should be only 1 frame long
            assert len(smpl_data["poses"]) == 1
            continue

        if False:
            new_smpl_data = smplrifkefeats_to_smpldata(features)
            new_features = smpldata_to_smplrifkefeats(new_smpl_data)
            newer_smpl_data = smplrifkefeats_to_smpldata(new_features)
            # consitent
            _val = 0.0
            _val += torch.abs(new_features - features).mean()
            _val += torch.abs(newer_smpl_data["trans"] - new_smpl_data["trans"]).mean()
            _val += torch.abs(newer_smpl_data["poses"] - new_smpl_data["poses"]).mean()
            _val += torch.abs(
                newer_smpl_data["joints"] - new_smpl_data["joints"]
            ).mean()

            assert _val < 1e-7

        features = features.numpy()

        # save
        np.save(new_motion_path, features)


def main():
    base_name = "AMASS_20.0_fps_nh"
    joints_name = "AMASS_20.0_fps_nh_smpljoints_neutral_nobetas"
    new_name = "AMASS_20.0_fps_nh_smplrifke"
    folder = "datasets/motions"
    force_redo = False

    get_features(folder, base_name, joints_name, new_name, force_redo=force_redo)


if __name__ == "__main__":
    main()
