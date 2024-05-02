import torch
import einops
from torch import Tensor

from geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)


def smpldata_to_smplrifkefeats(smpldata) -> Tensor:
    poses = smpldata["poses"].clone()
    trans = smpldata["trans"].clone()
    joints = smpldata["joints"].clone()

    # Sequence level
    assert poses.shape[-1] == 66  # 22 * 3 -> SMPL with no hands
    assert len(poses.shape) == 2
    assert len(trans.shape) == 2
    assert len(joints.shape) == 3

    ### JOINTS PROCESS

    # First remove the ground
    ground = joints[:, :, 2].min()
    joints[:, :, 2] -= ground

    root_grav_axis = joints[:, 0, 2].clone()

    # Make sure values are consitent
    _val = abs(
        (trans[:, 2] - trans[0, 2]) - (root_grav_axis - root_grav_axis[0])
    ).mean()
    assert _val < 1e-6

    # Trajectory => Translation without gravity axis (Z)
    trajectory = joints[:, 0, :2].clone()

    # Make sure values are consitent
    _val = torch.abs(
        (trajectory - trajectory[0]) - (trans[:, :2] - trans[0, :2])
    ).mean()
    assert _val < 1e-6

    # Joints in the pelvis coordinate system
    joints[:, :, [0, 1]] -= trajectory[..., None, :]
    # Also doing it for the Z coordinate
    joints[:, :, 2] -= joints[:, [0], 2]

    # check that the pelvis is all zero
    assert (joints[:, 0] == 0).all()

    # Delete the pelvis from the local representation
    # it is already encoded in root_grav_axis and trajectory
    joints = joints[:, 1:]

    vel_trajectory = torch.diff(trajectory, dim=0)
    # repeat the last acceleration
    # for the last (not seen) velocity
    last_acceleration = vel_trajectory[-1] - vel_trajectory[-2]
    future_velocity = vel_trajectory[-1] + last_acceleration
    vel_trajectory = torch.cat((vel_trajectory, future_velocity[None]), dim=0)

    ### SMPL PROCESS

    # unflatten
    poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
    poses_mat = axis_angle_to_matrix(poses)

    global_orient = poses_mat[:, 0]
    # Decompose the rotation into 3 euler angles rotations
    # To extract and remove the Z rotation for each frames
    global_euler = matrix_to_euler_angles(global_orient, "ZYX")
    rotZ_angle, rotY_angle, rotX_angle = torch.unbind(global_euler, -1)

    # Construct the rotations matrices
    rotZ = axis_angle_rotation("Z", rotZ_angle)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)

    # check the reconstruction
    global_orient_recons = rotZ @ rotY @ rotX
    # sanity check
    assert torch.abs(global_orient - global_orient_recons).mean() < 1e-6

    # construct the local global pose
    # the one without the final Z rotation
    global_orient_local = rotY @ rotX

    # True difference of angles
    # robust way of computing torch.diff with angles
    vel_rotZ = rotZ[1:] @ rotZ.transpose(1, 2)[:-1]
    # repeat the last acceleration (same as the trajectory but in the 3D rotation space)
    last_acc_rotZ = vel_rotZ[-1] @ vel_rotZ.transpose(1, 2)[-2]
    future_vel_rotZ = vel_rotZ[-1] @ last_acc_rotZ
    vel_rotZ = torch.cat((vel_rotZ, future_vel_rotZ[None]), dim=-3)
    vel_angles = matrix_to_axis_angle(vel_rotZ)[:, 2]

    # Rotate the vel_trajectory (rotation inverse in the indexes)
    vel_trajectory_local = torch.einsum("tkj,tk->tj", rotZ[:, :2, :2], vel_trajectory)
    # Rotate the local_joints
    joints_local = torch.einsum("tkj,tlk->tlj", rotZ[:, :2, :2], joints[:, :, [0, 1]])
    joints_local = torch.stack(
        (joints_local[..., 0], joints_local[..., 1], joints[..., 2]), axis=-1
    )

    # Replace the global orient with the one without rotation
    poses_mat_local = torch.cat((global_orient_local[:, None], poses_mat[:, 1:]), dim=1)
    poses_local = matrix_to_rotation_6d(poses_mat_local)

    # Stack things together
    features = group(
        root_grav_axis, vel_trajectory_local, vel_angles, poses_local, joints_local
    )
    return features


def smplrifkefeats_to_smpldata(features: Tensor):
    (
        root_grav_axis,
        vel_trajectory_local,
        vel_angles,
        poses_local,
        joints_local,
    ) = ungroup(features)

    poses_mat_local = rotation_6d_to_matrix(poses_local)
    global_orient_local = poses_mat_local[:, 0]

    # Remove the dummy last angle and integrate the angles
    angles = torch.cumsum(vel_angles[:-1], dim=0)
    # The first angle is zero (canonicalization)
    angles = torch.cat((0 * angles[[0]], angles), dim=0)

    # Construct the rotation matrix
    rotZ = axis_angle_rotation("Z", angles)

    # Rotate the trajectory (normal rotation in the indexes)
    vel_trajectory = torch.einsum("bjk,bk->bj", rotZ[:, :2, :2], vel_trajectory_local)

    joints = torch.einsum("bjk,blk->blj", rotZ[:, :2, :2], joints_local[..., [0, 1]])
    joints = torch.stack(
        (joints[..., 0], joints[..., 1], joints_local[..., 2]), axis=-1
    )

    # Remove the dummy last velocity and integrate the trajectory
    trajectory = torch.cumsum(vel_trajectory[..., :-1, :], dim=-2)
    # The first position is zero
    trajectory = torch.cat((0 * trajectory[..., [0], :], trajectory), dim=-2)

    # Add the pelvis (which is still zero)
    joints = torch.cat((0 * joints[:, [0]], joints), axis=1)

    # Adding back the Z component
    joints[:, :, 2] += root_grav_axis[:, None]
    # Adding back the trajectory
    joints[:, :, [0, 1]] += trajectory[:, None]

    # Get back the translation
    trans = torch.cat([trajectory, root_grav_axis[..., None]], dim=1)

    # Remove the predicted Z rotation inside global_orient_local
    # It is trained to be zero, but the network could produce non zeros outputs
    global_euler_local = matrix_to_euler_angles(global_orient_local, "ZYX")
    _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)

    # Replace it with the one computed with velocities
    global_orient = rotZ @ rotY @ rotX
    poses_mat = torch.cat(
        (global_orient[..., None, :, :], poses_mat_local[..., 1:, :, :]), dim=-3
    )

    poses = matrix_to_axis_angle(poses_mat)
    # flatten back
    poses = einops.rearrange(poses, "k l t -> k (l t)")
    smpldata = {"poses": poses, "trans": trans, "joints": joints}
    return smpldata


def group(root_grav_axis, vel_trajectory_local, vel_angles, poses_local, joints_local):
    # 1: root_grav_axis
    # 2: vel_trajectory_local
    # 1: vel_angles
    # 132 = 22*6: poses_local_flatten
    # 69 = 23*3: joints_local_flatten
    # total: 205

    # Flatten
    poses_local_flatten = einops.rearrange(poses_local, "k l t -> k (l t)")
    joints_local_flatten = einops.rearrange(joints_local, "k l t -> k (l t)")

    # Stack things together
    features, _ = einops.pack(
        [
            root_grav_axis,
            vel_trajectory_local,
            vel_angles,
            poses_local_flatten,
            joints_local_flatten,
        ],
        "k *",
    )

    assert features.shape[-1] == 205
    return features


def ungroup(features: Tensor) -> tuple[Tensor]:
    assert features.shape[-1] == 205
    (
        root_grav_axis,
        vel_trajectory_local,
        vel_angles,
        poses_local_flatten,
        joints_local_flatten,
    ) = einops.unpack(features, [[], [2], [], [132], [69]], "k *")

    poses_local = einops.rearrange(poses_local_flatten, "k (l t) -> k l t", t=6)
    joints_local = einops.rearrange(joints_local_flatten, "k (l t) -> k l t", t=3)
    return root_grav_axis, vel_trajectory_local, vel_angles, poses_local, joints_local


def canonicalize_rotation(smpldata):
    features = smpldata_to_smplrifkefeats(smpldata)
    return smplrifkefeats_to_smpldata(features)
