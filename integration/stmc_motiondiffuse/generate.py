# This code is adapted from https://github.com/mingyuan-zhang/MotionDiffuse/tree/main
import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from torch.utils.data import DataLoader
from utils.plot_script import *
from utils.get_opt import get_opt
from datasets.evaluator_models import MotionLenEstimatorBiGRU
from trainers import DDPMTrainer
from models import MotionTransformer
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.utils import *
from utils.motion_process import recover_from_ric
from collections import defaultdict

from stmc_motiondiffuse.stmc_wrapper import STMC_wrapper
from stmc_motiondiffuse.stmc import read_timelines, process_timelines

# for SINC baselines
from stmc_motiondiffuse.stmc import combine_features_intervals, interpolate_intervals


def plot_t2m(data, result_path, npy_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
    joint = motion_temporal_filter(joint, sigma=1)
    plot_3d_motion(
        result_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20
    )
    if npy_path != "":
        np.save(npy_path, joint)


def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
    )
    return encoder


if __name__ == "__main__":
    # WIP
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt_path",
        default="checkpoints/t2m/t2m_motiondiffuse/opt.txt",
        type=str,
        help="Opt path",
    )
    parser.add_argument(
        "--input_timeline",
        type=str,
        help="Path to a timeline file.",
    )
    parser.add_argument(
        "--stmc_baseline",
        choices=["none", "sinc", "sinc_lerp"],
        default="none",
        type=str,
        help="STMC baseline",
    )
    parser.add_argument(
        "--interval_overlap",
        default=0.5,
        type=float,
        help="Overlap (in seconds) for the diffcollage per body part",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="test_sample.gif",
        help="Path to save generation result",
    )
    parser.add_argument(
        "--npy_path", type=str, default="", help="Path to save 3D keypoints sequence"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    args = parser.parse_args()

    assert args.input_timeline
    assert os.path.exists(args.input_timeline)

    timeline_name = os.path.split(args.input_timeline)[-1]
    fps = 20.0
    timelines = read_timelines(args.input_timeline, fps)
    assert len(timelines) == 1
    # make sure it is a single timeline
    # otherwise we need to ask for a folder instead of a .npy save file

    interval_overlap = int(fps * args.interval_overlap)

    device = torch.device("cuda:%d" % args.gpu_id if args.gpu_id != -1 else "cpu")
    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True

    assert opt.dataset_name == "t2m"
    opt.data_root = "./dataset/HumanML3D"
    opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
    opt.text_dir = pjoin(opt.data_root, "texts")
    opt.joints_num = 22
    opt.dim_pose = 263
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    num_classes = 200 // opt.unit_length

    fps = 20.0
    num_repetitions = 1

    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))

    encoder = build_models(opt).to(device)
    trainer = DDPMTrainer(opt, encoder)
    trainer.load(pjoin(opt.model_dir, opt.which_epoch + ".tar"))

    trainer.eval_mode()
    trainer.to(opt.device)

    # Add STMC functionality
    STMC_wrapper(trainer, args.stmc_baseline)

    interval_overlap = int(fps * args.interval_overlap)

    with torch.no_grad():
        if "sinc" in args.stmc_baseline:
            # No extension and no unconditional transitions
            infos = process_timelines(
                timelines, interval_overlap, extend=False, uncond=False
            )
            infos["output_lengths"] = infos["max_t"]
            infos["featsname"] = "guoh3dfeats"
            m_lens = torch.LongTensor(infos["all_lengths"]).to(device)
            # lenghts of all the separate motions
        else:
            infos = process_timelines(timelines, interval_overlap)
            infos["output_lengths"] = infos["max_t"]
            infos["featsname"] = "guoh3dfeats"
            m_lens = torch.LongTensor(infos["max_t"]).to(device)
            # lenghts of the whole timelines

        lengths = infos["max_t"]
        infos["max_frames"] = 196
        caption = infos["all_texts"]

        motions = trainer.generate_batch(caption, m_lens, opt.dim_pose, infos)

        if "sinc" in args.stmc_baseline:
            # regroup the motions into timelines
            nfeats = motions.shape[-1]
            shape = (len(infos["timeline"]), max(infos["max_t"]), nfeats)
            output = motions.new_zeros(shape)

            motions = combine_features_intervals(motions, infos, output)

        if "lerp" in args.stmc_baseline or "interp" in args.stmc_baseline:
            # interpolate to smooth the results
            motions = interpolate_intervals(motions, infos)

        motions = motions.cpu().numpy()
        motions = motions * std + mean

        assert len(motions) == 1
        motion = motions[0, : infos["max_t"][0]]

        title = timeline_name
        plot_t2m(motion, args.result_path, args.npy_path, title)
