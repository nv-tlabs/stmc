import numpy as np


def extract_joints(x, featsname, **kwargs):
    if featsname == "smplrifke":
        return extract_joints_smplrifke(x, **kwargs)
    elif featsname == "guoh3dfeats":
        return extract_joints_guoh3dfeats(x, **kwargs)
    else:
        raise NotImplementedError


def extract_joints_smplrifke(
    x, fps, value_from="joints", smpl_layer=None, first_angle=np.pi, **kwargs
):
    assert x.shape[-1] == 205
    if value_from == "smpl":
        assert smpl_layer is not None

    # smplrifke
    from src.tools.smplrifke_feats import smplrifkefeats_to_smpldata

    smpldata = smplrifkefeats_to_smpldata(x, first_angle=first_angle)

    smpldata["mocap_framerate"] = fps
    poses = smpldata["poses"]
    trans = smpldata["trans"]
    joints = smpldata["joints"]

    if value_from == "smpl":
        vertices, joints = smpl_layer(poses, trans)
        output = {
            "vertices": vertices.numpy(),
            "joints": joints.numpy(),
            "smpldata": smpldata,
        }
    elif value_from == "joints":
        output = {"joints": joints.numpy()}
    else:
        raise NotImplementedError
    return output


def extract_joints_guoh3dfeats(x, **kwargs):
    assert x.shape[-1] == 263
    from src.tools.guofeats import guofeats_to_joints

    joints = guofeats_to_joints(x).numpy()
    output = {"joints": joints.numpy()}
    return output
