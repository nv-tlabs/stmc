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
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil

# STMC specific
from stmc_mdm.parser_stmc import generate_args
from stmc_mdm.stmc_wrapper import STMC_wrapper
from stmc_mdm.stmc import read_timelines, process_timelines

# for SINC baselines
from stmc_mdm.stmc import combine_features_intervals, interpolate_intervals


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if args.dataset == "kit" else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    if not args.input_timeline:
        raise Exception(
            "To condition on a timeline, please use the --input_timeline argument."
        )

    assert os.path.exists(args.input_timeline)
    assert args.stmc_baseline in ["none", "sinc", "sinc_lerp"]

    dist_util.setup_dist(args.device)
    if out_path == "":
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            "samples_{}_{}_seed{}".format(name, niter, args.seed),
        )

        out_path += "_" + os.path.basename(args.input_timeline).replace(
            ".txt", "_timeline"
        ).replace(" ", "_").replace(".", "")

        if args.stmc_baseline != "none":
            # baselines use another folder
            out_path = out_path.replace(
                "_timeline", "_timeline_baseline_" + args.stmc_baseline
            )

    timelines = read_timelines(args.input_timeline, fps)
    interval_overlap = int(fps * args.interval_overlap)

    # Actually mask and lengths are never used in MDM
    # so no need to fill the "model_kwargs" dict.
    # (MDM always deals with 196 frames motions).

    if "sinc" in args.stmc_baseline:
        # No extension and no unconditional transitions for SINC
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
        n_samples = len(infos["max_t"])

    model_kwargs = {"y": y_dict}

    lengths = infos["max_t"]
    args.num_samples = len(lengths)
    args.batch_size = args.num_samples

    print("Loading dataset...")
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

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

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f"### Sampling [repetitions #{rep_i}]")

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(n_samples, device=dist_util.dev()) * args.guidance_param
            )

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
            (n_samples, model.njoints, model.nfeats, MAX_FRAMES),  # BUG FIX
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
                len(infos["timeline"]),
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
            else model_kwargs["y"]["mask"].reshape(args.batch_size, n_frames).bool()
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

        model_kwargs["y"]["lengths"] = torch.from_numpy(np.array(infos["max_t"]))
        all_text += ["_".join([text for text in texts]) for texts in infos["texts"]]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "num_samples": args.num_samples,
            "num_repetitions": args.num_repetitions,
        },
    )
    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = (
        paramUtil.kit_kinematic_chain
        if args.dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )

    sample_files = []
    num_samples_in_out_file = 7

    (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    ) = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            length = all_lengths[rep_i * args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[
                :length
            ]
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(
                animation_save_path,
                skeleton,
                motion,
                dataset=args.dataset,
                title="",
                fps=fps,
            )
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(
            args,
            out_path,
            row_print_template,
            all_print_template,
            row_file_template,
            all_file_template,
            caption,
            num_samples_in_out_file,
            rep_files,
            sample_files,
            sample_i,
        )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


def save_multiple_samples(
    args,
    out_path,
    row_print_template,
    all_print_template,
    row_file_template,
    all_file_template,
    caption,
    num_samples_in_out_file,
    rep_files,
    sample_files,
    sample_i,
):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f" -i {f} " for f in rep_files]
    hstack_args = (
        f" -filter_complex hstack=inputs={args.num_repetitions}"
        if args.num_repetitions > 1
        else ""
    )
    ffmpeg_rep_cmd = (
        f"ffmpeg -y -loglevel warning "
        + "".join(ffmpeg_rep_files)
        + f"{hstack_args} {all_rep_save_path}"
    )
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (
        sample_i + 1
    ) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(
            sample_i - len(sample_files) + 1, sample_i
        )
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(
            all_print_template.format(
                sample_i - len(sample_files) + 1, sample_i, all_sample_save_file
            )
        )
        ffmpeg_rep_files = [f" -i {f} " for f in sample_files]
        vstack_args = (
            f" -filter_complex vstack=inputs={len(sample_files)}"
            if len(sample_files) > 1
            else ""
        )
        ffmpeg_rep_cmd = (
            f"ffmpeg -y -loglevel warning "
            + "".join(ffmpeg_rep_files)
            + f"{vstack_args} {all_sample_save_path}"
        )
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = "sample{:02d}.mp4"
    all_file_template = "samples_{:02d}_to_{:02d}.mp4"
    if unconstrained:
        sample_file_template = "row{:02d}_col{:02d}.mp4"
        sample_print_template = "[{} row #{:02d} column #{:02d} | -> {}]"
        row_file_template = row_file_template.replace("sample", "row")
        row_print_template = "[{} row #{:02d} | all columns | -> {}]"
        all_file_template = all_file_template.replace("samples", "rows")
        all_print_template = "[rows {:02d} to {:02d} | -> {}]"
    else:
        sample_file_template = "sample{:02d}_rep{:02d}.mp4"
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = "[samples {:02d} to {:02d} | all repetitions | -> {}]"

    return (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    )


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
