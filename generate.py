import logging
import hydra

from hydra.utils import instantiate
from omegaconf import DictConfig
from src.config import read_config

import os
import shutil


# avoid conflic between tokenizer and rendering
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="generate", version_base="1.3")
def generate(c: DictConfig):
    logger.info("Prediction script")

    assert c.input_type in ["text", "timeline", "auto"]
    assert c.baseline in ["none", "sinc", "sinc_lerp"]

    exp_folder_name = os.path.splitext(os.path.split(c.timeline)[-1])[0]

    if c.baseline != "none":
        exp_folder_name += "_baseline_" + c.baseline

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps

    interval_overlap = int(fps * c.overlap_s)

    from src.stmc import read_timelines, process_timelines
    from src.text import read_texts

    if c.input_type == "auto" or "timeline":
        try:
            timelines = read_timelines(c.timeline, fps)
            logger.info("Reading the timelines")
            n_motions = len(timelines)
            c.input_type = "timeline"
        except IndexError:
            c.input_type = "text"
    if c.input_type == "text":
        logger.info("Reading the texts")
        texts_durations = read_texts(c.timeline, fps)
        n_motions = len(texts_durations)

    logger.info("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np
    import torch

    if c.input_type == "text":
        infos = {
            "texts_durations": texts_durations,
            "all_lengths": [x.duration for x in texts_durations],
            "all_texts": [x.text for x in texts_durations],
        }
        infos["output_lengths"] = infos["all_lengths"]
    elif c.input_type == "timeline":
        infos = process_timelines(timelines, interval_overlap)
        infos["output_lengths"] = infos["max_t"]

        if c.baseline != "none":
            infos["baseline"] = c.baseline

    infos["featsname"] = cfg.motion_features
    infos["guidance_weight"] = c.guidance

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    logger.info("Loading the checkpoint")

    ckpt = torch.load(ckpt_path, map_location=c.device)
    # Models
    logger.info("Loading the models")

    # Rendering
    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)

    # Diffusion model
    # update the folder first, in case it has been moved
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(c.run_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(c.run_dir, "text_stats")

    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])

    # Evaluation mode
    diffusion.eval()
    diffusion.to(c.device)

    # jointstype = "smpljoints"
    jointstype = "both"

    from src.tools.smpl_layer import SMPLH

    smplh = SMPLH(
        path="deps/smplh",
        jointstype=jointstype,
        input_pose_rep="axisangle",
        gender=c.gender,
    )

    from src.model.text_encoder import TextToEmb

    modelpath = cfg.data.text_encoder.modelname
    mean_pooling = cfg.data.text_encoder.mean_pooling
    text_model = TextToEmb(
        modelpath=modelpath, mean_pooling=mean_pooling, device=c.device
    )

    logger.info("Generate the function")

    video_dir = os.path.join(
        c.run_dir,
        "generations",
        exp_folder_name + "_" + str(ckpt_name) + f"_{c.input_type}_to_motion",
    )
    os.makedirs(video_dir, exist_ok=True)

    shutil.copy(
        c.timeline, os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}.txt")
    )

    vext = ".mp4"

    joints_video_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}_joints{vext}")
        for idx in range(n_motions)
    ]

    smpl_video_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}_smpl{vext}")
        for idx in range(n_motions)
    ]

    npy_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}.npy")
        for idx in range(n_motions)
    ]

    logger.info(f"All the output videos will be saved in: {video_dir}")

    if c.seed != -1:
        pl.seed_everything(c.seed)

    with torch.no_grad():
        tx_emb = text_model(infos["all_texts"])
        tx_emb_uncond = text_model(["" for _ in infos["all_texts"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(c.device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(
                    c.device
                ),
            }

        xstarts = diffusion(tx_emb, tx_emb_uncond, infos).cpu()

        for idx, (xstart, length) in enumerate(zip(xstarts, infos["output_lengths"])):
            xstart = xstart[:length]

            from src.tools.extract_joints import extract_joints

            output = extract_joints(
                xstart,
                infos["featsname"],
                fps=fps,
                value_from=c.value_from,
                smpl_layer=smplh,
            )

            joints = output["joints"]
            path = npy_paths[idx]
            np.save(path, joints)

            if "vertices" in output:
                path = npy_paths[idx].replace(".npy", "_verts.npy")
                np.save(path, output["vertices"])

            if "smpldata" in output:
                path = npy_paths[idx].replace(".npy", "_smpl.npz")
                np.savez(path, **output["smpldata"])

            logger.info(f"Joints rendering {idx}")
            joints_renderer(
                joints, title="", output=joints_video_paths[idx], canonicalize=False
            )
            print(joints_video_paths[idx])
            print()

            if "vertices" in output and not c.fast:
                logger.info(f"SMPL rendering {idx}")
                smpl_renderer(
                    output["vertices"], title="", output=smpl_video_paths[idx]
                )
                print(smpl_video_paths[idx])
                print()

            logger.info("Rendering done")


if __name__ == "__main__":
    generate()
