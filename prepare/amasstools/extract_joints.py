import os
import argparse
import numpy as np

import torch
from smplh_layer import load_smplh_gender
from loop_amass import loop_amams


def extract_joints(
    base_folder,
    new_base_folder,
    smplh_folder,
    jointstype,
    batch_size,
    gender,
    use_betas,
    device,
    force_redo,
):
    print(
        "Extract joint position ({}) from SMPL pose parameter, {} betas and {}".format(
            jointstype,
            "with" if use_betas else "without",
            "with {gender} body shape"
            if gender != "gendered"
            else "with original gender",
        )
    )
    print("The processed motions will be stored in this folder:")
    print(new_base_folder)

    smplh = load_smplh_gender(gender, smplh_folder, jointstype, batch_size, device)

    iterator = loop_amams(
        base_folder,
        new_base_folder,
        ext=".npz",
        newext=".npy",
        force_redo=force_redo,
    )

    for motion_path, new_motion_path in iterator:
        data = np.load(motion_path)

        # process sequences
        poses = torch.from_numpy(data["poses"]).to(torch.float).to(device)
        trans = torch.from_numpy(data["trans"]).to(torch.float).to(device)

        if use_betas and "betas" in data and data["betas"] is not None:
            betas = torch.from_numpy(data["betas"]).to(torch.float).to(device)
        else:
            betas = None

        if gender == "gendered":
            gender_motion = str(data["gender"])
            smplh_layer = smplh[gender_motion]
        else:
            smplh_layer = smplh

        joints = smplh_layer(poses, trans, betas).cpu().numpy()
        np.save(new_motion_path, joints)


def main():
    base_folder = "datasets/motions/AMASS_20.0_fps_nh"
    smplh_folder = "deps/smplh"
    jointstype = "smpljoints"
    batch_size = 4096
    gender = "neutral"
    use_betas = False
    force_redo = False

    name = os.path.split(base_folder)[1]
    new_base_folder = f"datasets/motions/{name}_{jointstype}_{gender}_{'betas' if use_betas else 'nobetas'}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extract_joints(
        base_folder,
        new_base_folder,
        smplh_folder,
        jointstype,
        batch_size,
        gender,
        use_betas,
        device,
        force_redo,
    )


if __name__ == "__main__":
    main()
