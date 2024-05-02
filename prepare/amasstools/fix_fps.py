import numpy as np

import torch
from einops import rearrange, repeat

from slerp import slerp
from loop_amass import loop_amams


def interpolate_fps_joints(joints, old_fps, new_fps, mode="linear"):
    assert old_fps != 0
    scale_factor = new_fps / old_fps

    # joints: [..., T, J, 3]
    joints = joints.swapaxes(-3, -1)
    # joints: [..., 3, J, T]
    joints = torch.nn.functional.interpolate(
        joints, scale_factor=scale_factor, mode=mode
    )
    # joints: [..., 3, J, T2]
    joints = joints.swapaxes(-3, -1)
    # joints: [..., T2, J, 3]
    return joints


def interpolate_fps_trans(trans, old_fps, new_fps, mode="linear"):
    joints = trans[:, None]
    inter_joints = interpolate_fps_joints(joints, old_fps, new_fps, mode=mode)
    return inter_joints[:, 0]


def interpolate_fps_poses(poses, old_fps, new_fps, mode="linear"):
    # slerp interpolation
    assert old_fps != 0
    scale_factor = new_fps / old_fps

    # Get back axis angle dimension
    poses = rearrange(poses, "i (k t) -> i k t", t=3)

    # Find the interpolation indices
    nframes = len(poses)
    indices = torch.arange(nframes)
    inter_indices = torch.nn.functional.interpolate(
        indices.to(torch.float)[None, None], scale_factor=scale_factor, mode=mode
    )[0, 0]

    floor = torch.floor(inter_indices).to(int)
    ceil = torch.ceil(inter_indices).to(int)
    fraction_part = inter_indices - floor

    v0 = poses[floor]
    v1 = poses[ceil]
    t = repeat(fraction_part, "x -> x r k", r=v0.shape[1], k=v0.shape[2])
    # Interpolate inbetween with slerp
    interpolated_poses = slerp(v0, v1, t)
    interpolated_poses = rearrange(interpolated_poses, "i k t -> i (k t)")
    return interpolated_poses


def fix_fps(base_folder, new_base_folder, new_fps, force_redo=False):
    print(f"Fix the fps to {new_fps}")
    print("The processed motions will be stored in this folder:")
    print(new_base_folder)

    iterator = loop_amams(
        base_folder, new_base_folder, ext=".npz", newext=".npz", force_redo=force_redo
    )

    for motion_path, new_motion_path in iterator:
        if "humanact12" in motion_path:
            continue

        data = {x: y for x, y in np.load(motion_path).items()}

        old_fps = float(data["mocap_framerate"])

        # process sequences
        poses = torch.from_numpy(data["poses"]).to(torch.float)
        trans = torch.from_numpy(data["trans"]).to(torch.float)

        # remove hands / which are almost always the mean in AMASS
        poses = poses[:, :66]

        try:
            inter_poses = interpolate_fps_poses(poses, old_fps, new_fps).numpy()
            inter_trans = interpolate_fps_trans(trans, old_fps, new_fps).numpy()
        except RuntimeError:
            # The sequence should be only 1 frame long
            assert len(trans) == 1
            # In this case return the original data
            inter_poses = poses
            inter_trans = trans

        new_data = data.copy()
        new_data.update(
            {
                "trans": inter_trans,
                "poses": inter_poses,
                "mocap_framerate": new_fps,
            }
        )

        np.savez(new_motion_path, **new_data)


def main():
    new_fps = 20.0

    force_redo = False
    base_folder = "datasets/motions/AMASS"
    new_base_folder = f"datasets/motions/AMASS_{new_fps}_fps_nh"  # nh for no hands

    fix_fps(base_folder, new_base_folder, new_fps, force_redo)


if __name__ == "__main__":
    main()
