import os
import orjson
import numpy as np

from src.guofeats import joints_to_guofeats

AMASS_FOLDER = "../datasets/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas/"
TEXT_PATH = "mtt/texts.txt"
MOTION_PATH = "mtt/motions.txt"


def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


def load_lines(path):
    with open(path) as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


def load_texts(path):
    lines = load_lines(path)
    texts = [l.split("#")[0].strip() for l in lines]
    return texts


def load_motions(amass_folder, path, fps):
    lines = load_lines(path)
    amass_paths = [l.split("#")[0].strip() for l in lines]
    begins = [int(fps * float(l.split("#")[1].strip())) for l in lines]
    ends = [int(fps * float(l.split("#")[2].strip())) for l in lines]

    motions = []
    for amass_path, begin, end in zip(amass_paths, begins, ends):
        path = os.path.join(amass_folder, amass_path + ".npy")
        motion = np.load(path)
        motion = motion[begin:end]
        motions.append(motion)
    return motions


def load_mtt_texts_motions(fps):
    texts = load_texts(TEXT_PATH)
    motions = load_motions(AMASS_FOLDER, MOTION_PATH, fps)
    # convert to guofeats
    # first, make sure to revert the axis
    # as guofeats have gravity axis in Y
    feats = []
    for motion in motions:
        x, y, z = motion.T
        joint = np.stack((x, z, -y), axis=0).T
        feats.append(joints_to_guofeats(joint))

    return texts, feats
