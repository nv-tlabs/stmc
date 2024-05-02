# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# which is based on https://github.com/openai/guided-diffusion
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric

# STMC specific
from stmc_mdm.parser_stmc import generate_args
from stmc_mdm.stmc_wrapper import STMC_wrapper
from stmc_mdm.stmc import read_timelines, process_timelines

# for SINC baselines
from stmc_mdm.stmc import combine_features_intervals, interpolate_intervals

import einops


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    assert args.dataset == "humanml"

    assert args.stmc_baseline in ["none", "sinc", "sinc_lerp", "singletrack", "onetext"]

    fps = 12.5 if args.dataset == "kit" else 20
    interval_overlap = int(fps * args.interval_overlap)

    mtt_name = "mtt"
    if args.stmc_baseline == "onetext":
        mtt_file = "mtt/baselines/MTT_onetext.txt"
    elif args.stmc_baseline == "singletrack":
        mtt_file = "mtt/baselines/MTT_singletrack.txt"
    else:
        mtt_file = "mtt/MTT.txt"

    print("Reading the timelines")
    all_timelines = read_timelines(mtt_file, fps)
    n_sequences = len(all_timelines)

    n_frames = min(max_frames, int(args.motion_length * fps))
    dist_util.setup_dist(args.device)

    print("Loading dataset...")
    # n_frames does not matter
    # we don't use the data
    data = load_dataset(args, max_frames, n_frames)
    # total_num_samples = args.num_samples * num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    STMC_wrapper(model, diffusion)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    out_path = os.path.join(
        os.path.dirname(args.model_path),
        "samples_{}_{}_seed{}".format(name, niter, args.seed),
    )

    out_path += "_" + mtt_name + "_timeline"

    if args.stmc_baseline != "none":
        # baselines use another folder
        out_path = out_path.replace(
            "_timeline", "_timeline_baseline_" + args.stmc_baseline
        )

    if args.interval_overlap != 0.5:
        out_path += "_intervaloverlap_" + str(args.interval_overlap)

    print(f"Saving the results here: {out_path}")
    os.makedirs(out_path, exist_ok=True)

    at_a_time = 50
    iterator = np.array_split(np.arange(n_sequences), n_sequences // at_a_time)

    with torch.no_grad():
        for x in iterator:
            timelines = [all_timelines[y] for y in x]
            npy_paths = [os.path.join(out_path, str(y).zfill(4) + ".npy") for y in x]
            n_seq_here = len(npy_paths)

            if "sinc" in args.stmc_baseline:
                # No extension and no unconditional transitions
                infos = process_timelines(
                    timelines, interval_overlap, extend=False, uncond=False
                )
                infos["output_lengths"] = infos["max_t"]
                infos["featsname"] = "guoh3dfeats"
                y_dict = {
                    "text": infos["all_texts"],
                    "lengths": torch.tensor(infos["all_lengths"]).to(dist_util.dev()),
                }
                n_samples = len(infos["all_texts"])
            else:
                infos = process_timelines(timelines, interval_overlap)
                infos["output_lengths"] = infos["max_t"]
                infos["featsname"] = "guoh3dfeats"
                y_dict = {"stmc_infos": infos, "max_frames": max_frames}
                n_samples = n_seq_here

            model_kwargs = {"y": y_dict}
            # int_mat_t = model_kwargs["y"]["interval"]["max_t"]

            if args.guidance_param != 1:
                model_kwargs["y"]["scale"] = (
                    torch.ones(n_samples, device=dist_util.dev()) * args.guidance_param
                )

            lengths = infos["max_t"]

            sample_fn = diffusion.p_sample_loop

            # Make sure to have enough space for STMC
            # This is because each mdm generation need to be exactly 196 frames:
            # there is no mask in the Transformer decoder
            # so it is always taking the whole input.
            MAX_FRAMES = max(lengths) + max_frames
            if "sinc" in args.stmc_baseline:
                MAX_FRAMES = max_frames  # = 196
                # is what we want for the sinc baseline
                # as the forward passes are independants

            sample = sample_fn(
                model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (
                    n_samples,
                    model.njoints,
                    model.nfeats,
                    MAX_FRAMES,
                ),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            if "sinc" in args.stmc_baseline:
                shape = (
                    len(infos["max_t"]),
                    *sample[0].shape[:-1],
                    max(infos["max_t"]),
                )
                output = sample[0].new_zeros(shape)
                # combine the output with SINC
                sample = combine_features_intervals(sample, infos, output)

                if "lerp" in args.stmc_baseline or "interp" in args.stmc_baseline:
                    # interpolate to smooth the results
                    sample = interpolate_intervals(sample, infos)

            # Recover XYZ *positions* from HumanML3D vector representation
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = (
                "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
            )

            rot2xyz_mask = (
                None
                if rot2xyz_pose_rep == "xyz"
                else model_kwargs["y"]["mask"].reshape(n_seq_here, n_frames).bool()
            )
            sample = model.rot2xyz(
                x=sample,
                mask=rot2xyz_mask,
                pose_rep=rot2xyz_pose_rep,
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )

            sample = einops.rearrange(sample, "i j f t -> i t j f")
            sample = sample.detach().cpu().numpy()
            for idx, (length, npy_path) in enumerate(zip(infos["max_t"], npy_paths)):
                unic_sample = sample[idx, :length]
                # shape T, 22, 3
                np.save(npy_path, unic_sample)


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",
    )
    if args.dataset in ["kit", "humanml"]:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
