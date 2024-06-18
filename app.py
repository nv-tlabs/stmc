# Inpired by the TMR repo:
# https://github.com/Mathux/TMR

import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.config import read_config
import pyrender  # noqa

import time
import os

import torch
import gradio as gr
from gradio.themes.utils import sizes

import pytorch_lightning as pl

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from app_config import CONFIG, FPS, SHARE
from src.stmc import process_timelines, TextInterval
from src.model.text_encoder import TextToEmb
from src.tools.smpl_layer import SMPLH
from src.tools.extract_joints import extract_joints
import src.prepare  # noqa
import numpy as np

# avoid conflic between tokenizer and rendering
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

config = CONFIG.copy()
fps = FPS
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

c = OmegaConf.create(config)
cfg = read_config(c.run_dir)

WEBSITE = """
<div class="embed_hidden">
<h1> STMC: Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation</h1>

<h2>
<a href="https://mathis.petrovich.fr" target="_blank"><nobr>Mathis Petrovich</nobr></a> &emsp;
<a href="https://orlitany.github.io" target="_blank"><nobr>Or Litany</nobr></a> &emsp;
<a href="https://www.umariqbal.info" target="_blank"><nobr>Umar Iqbal</nobr></a> &emsp;
<a href="https://ps.is.mpg.de/~black" target="_blank"><nobr>Michael J. Black</nobr></a> &emsp;
<a href="https://imagine.enpc.fr/~varolg" target="_blank"><nobr>G&uumll Varol</nobr></a> &emsp;
<a href="https://xbpeng.github.io" target="_blank"><nobr>Xue Bin Peng</nobr></a> &emsp;
<a href="https://davrempe.github.io" target="_blank"><nobr>Davis Rempe</nobr></a>
</h2>

<h2 style='text-align: center'; display:block>
<nobr>CVPRW 2024</nobr>
</h2>

<h3 class="centered" >
<a target="_blank" href="https://arxiv.org/abs/2401.08559"> <button type="button" class="btn btn-primary btn-lg"> Paper </button></a>
<a target="_blank" href="https://github.com/nv-tlabs/stmc"> <button type="button" class="btn btn-primary btn-lg"> Code </button></a>
<a target="_blank" href="https://mathis.petrovich.fr/stmc"> <button type="button" class="btn btn-primary btn-lg"> Webpage </button></a>
<a target="_blank" href="https://mathis.petrovich.fr/stmc/stmc.bib"> <button type="button" class="btn btn-primary btn-lg"> BibTex </button></a>
</h3>

<h3> Description </h3>
<p>
This space illustrates <a href='https://mathis.petrovich.fr/stmc/' target='_blank'><b>STMC</b></a>, a method for multi-track timeline control for text-driven 3D human motion generation. STMC is compatible with any diffusion model trained to denoised in the motion space. We train our own diffusion model (called MDM-SMPL), which directly outputs SMPL pose parameters.
</p>

<h3> Note </h3>
<p>
Our model is trained on the HumanML3D dataset, which typically contains complete sentences (e.g. "A person is sitting"). If some generations don't work very well, try using full sentences instead of keywords (e.g. "sitting down").
</p>
</div>

<h3> Generation </h3>
"""

MAX_TEXT = 10
DEFAULT_TEXTS = [
    "walking in a circle clockwise",
    "a person is jumping",
    "raising the right hand",
]
DEFAULT_BODY_PARTS = [["legs"], ["legs", "spine"], ["right arm"]]
DEFAULT_STARTS = [0.0, 5.0, 2.0]
DEFAULT_ENDS = [5.0, 8.0, 7.0]
DEFAULT_MAX_S = max(DEFAULT_ENDS)

DEFAULT_NTEXT = len(DEFAULT_TEXTS)
for x in range(MAX_TEXT - len(DEFAULT_TEXTS)):
    DEFAULT_TEXTS.append("")
    DEFAULT_STARTS.append(0.0)
    DEFAULT_ENDS.append(0.0)
    DEFAULT_BODY_PARTS.append([])


choices = ["left arm", "right arm", "legs", "head", "spine"]


# colors = ["blue", "green", "red", "black", "yellow", "purple", "gray", ""]
COLORS = [
    "royalblue",
    "hotpink",
    "palegreen",
    "lightcyan",
    "wheat",
    "lavender",
    "salmon",
    "silver",
    "lime",
    "plum",
]


def plot_box(seconds, number_of_texts, *args):
    if number_of_texts == 0:
        return None

    max_height = 10
    begins = args[:MAX_TEXT][:number_of_texts]
    ends = args[MAX_TEXT : 2 * MAX_TEXT][:number_of_texts]
    texts = args[2 * MAX_TEXT : 3 * MAX_TEXT][:number_of_texts]
    # body_parts = args[3 * MAX_TEXT : 4 * MAX_TEXT][:number_of_texts]

    # fig = plt.figure()
    fig, ax = plt.subplots()
    for idx, (begin, end, text) in enumerate(zip(begins, ends, texts)):
        width = end - begin
        height = max_height / number_of_texts
        rectangle = Rectangle(
            (begin, max_height - height * (idx + 1)),
            width,
            height,
            facecolor=COLORS[idx],
        )

        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width() / 2.0
        cy = ry + rectangle.get_height() / 2.0
        ax.annotate(
            text,
            (cx, cy),
            color="black",
            weight="bold",
            fontsize=10,
            ha="center",
            va="center",
        )

    plt.yticks([])
    plt.xlim([0, seconds])
    plt.ylim([0, max_height])
    plt.close()
    return fig


def update_numbers(number, seconds, *args):
    begins = args[:MAX_TEXT]
    ends = args[MAX_TEXT : 2 * MAX_TEXT]

    n_vis = range(number)
    n_novis = range(MAX_TEXT - number)

    update_begins = [
        gr.update(visible=True, maximum=seconds, value=min(begins[idx], seconds))
        for idx in n_vis
    ] + [gr.update(visible=False, maximum=seconds, value=0.0) for idx in n_novis]
    update_ends = [
        gr.update(visible=True, maximum=seconds, value=min(ends[idx], seconds))
        for idx in n_vis
    ] + [gr.update(visible=False, maximum=seconds, value=seconds) for idx in n_novis]

    update_texts = [gr.update(visible=True) for _ in n_vis] + [
        gr.update(visible=False) for _ in n_novis
    ]

    body_parts = [gr.update(visible=True) for _ in n_vis] + [
        gr.update(visible=False) for _ in n_novis
    ]
    return update_begins + update_ends + update_texts + body_parts


logger = logging.getLogger(__name__)


def render_process(val):
    i, motion, renderer, video_path_template, input_text = val
    video_path = video_path_template.format(i)
    return video_path


# Models
logger.info("Loading the model")
ckpt_name = c.ckpt
ckpt_path = os.path.join(c.run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
logger.info("Loading the checkpoint")
ckpt = torch.load(ckpt_path, map_location=device)
# Diffusion model
# update the folder first, in case it has been moved
cfg.diffusion.motion_normalizer.base_dir = os.path.join(c.run_dir, "motion_stats")
cfg.diffusion.text_normalizer.base_dir = os.path.join(c.run_dir, "text_stats")
diffusion = instantiate(cfg.diffusion)
diffusion.load_state_dict(ckpt["state_dict"])
# Evaluation mode
diffusion.eval()
diffusion.to(device)

smplh_dict = {
    gender: SMPLH(
        path="deps/smplh",
        jointstype="both",
        input_pose_rep="axisangle",
        gender=gender,
    )
    for gender in ["neutral", "male", "female"]
}

# Rendering
joints_renderer = instantiate(c.joints_renderer)
smpl_renderer = instantiate(c.smpl_renderer)

modelpath = cfg.data.text_encoder.modelname
mean_pooling = cfg.data.text_encoder.mean_pooling
text_model = TextToEmb(modelpath=modelpath, mean_pooling=mean_pooling, device=device)

video_dir = os.path.join(c.run_dir, f"app_generations_{str(ckpt_name)}")
os.makedirs(video_dir, exist_ok=True)
vext = ".mp4"


def generate(
    seconds,
    number_of_texts,
    guidance,
    overlap_s,
    seed,
    *args,
    progress=gr.Progress(track_tqdm=True),
):
    logger.info(f"Generate with {number_of_texts}")

    interval_overlap = int(fps * overlap_s)

    begins = list(args[:MAX_TEXT][:number_of_texts])
    ends = list(args[MAX_TEXT : 2 * MAX_TEXT][:number_of_texts])
    texts = list(args[2 * MAX_TEXT : 3 * MAX_TEXT][:number_of_texts])
    body_parts = list(args[3 * MAX_TEXT : 4 * MAX_TEXT][:number_of_texts])

    # start the motion at time 0
    min_t = min([x for x in begins])
    if min_t != 0.0:
        begins = [x - min_t for x in begins]
        ends = [x - min_t for x in ends]

    # find name of the generation
    name = f"app_gen_{overlap_s}_"
    for begin, end, text in zip(begins, ends, texts):
        text = text.replace(" ", "_")
        name += f"{text[:20]}_{begin}_{end}_"
    name += f"seed_{seed}"

    # Contruct the timelines objects
    timeline = [
        TextInterval(text, int(fps * begin), int(fps * end), bodyparts)
        for text, begin, end, bodyparts in zip(texts, begins, ends, body_parts)
    ]
    # only one timeline
    timelines = [timeline]

    # process the timelines
    infos = process_timelines(timelines, interval_overlap)
    infos["output_lengths"] = infos["max_t"]
    infos["featsname"] = cfg.motion_features
    infos["guidance_weight"] = guidance

    if seed != -1:
        pl.seed_everything(seed)

    joints_video_path = os.path.join(video_dir, name + "_joints.mp4")
    logger.info(f"Will be saved in: {joints_video_path}")

    with torch.no_grad():
        tx_emb = text_model(infos["all_texts"])
        tx_emb_uncond = text_model(["" for _ in infos["all_texts"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(
                    device
                ),
            }

        # diffusion
        xstarts = diffusion(
            tx_emb, tx_emb_uncond, infos, progress_bar=progress.tqdm
        ).cpu()
        # only one sample
        xstart = xstarts[0]
        length = infos["output_lengths"][0]
        xstart = xstart[:length]

        # npy paths saving
        smpl_rifke_feats_path = os.path.join(video_dir, name + "_smplrifke.npy")
        joints_path = os.path.join(video_dir, name + "_joints.npy")

        logger.info("Saving smplrifke features")
        np.save(smpl_rifke_feats_path, xstart)

        output = extract_joints(
            xstart,
            cfg.motion_features,
            fps=fps,
            value_from="joints",
            smpl_layer=None,
        )

        joints = output["joints"]
        np.save(joints_path, joints)

        logger.info("Joints rendering")
        progress(0, desc="Joints rendering...")
        joints_renderer(joints, title="", output=joints_video_path, canonicalize=False)
        progress(1, desc="Joints rendering...")
        time.sleep(0.5)
    return joints_video_path, smpl_rifke_feats_path


def get_data(smpl_rifke_feats_path, gender):
    try:
        xstart = torch.from_numpy(np.load(smpl_rifke_feats_path))
    except FileNotFoundError:
        return None, None, None, None, None
    output = extract_joints(
        xstart,
        cfg.motion_features,
        fps=fps,
        value_from="smpl",
        smpl_layer=smplh_dict[gender],
    )

    # joints path
    joints_path = smpl_rifke_feats_path.replace("_smplrifke.npy", "_joints.npy")

    # save the vertices in this path
    vertices_path = smpl_rifke_feats_path.replace("_smplrifke.npy", "_smpl.npy")
    np.save(vertices_path, output["vertices"])

    # save the smpl pose parameters there
    smpl_path = smpl_rifke_feats_path.replace("_smplrifke.npy", "_smpl.npz")
    np.savez(smpl_path, **output["smpldata"])

    return vertices_path, smpl_rifke_feats_path, joints_path, vertices_path, smpl_path


# separte function to not have to wait to much
def render_smpl(vertices_path, gender, progress=gr.Progress(track_tqdm=True)):
    # render the video to this path
    smpl_video_path = vertices_path.replace("_smpl.npy", "_smpl.mp4")

    try:
        vertices = np.load(vertices_path)
    except FileNotFoundError:
        return None
    logger.info("SMPL rendering")
    smpl_renderer(
        vertices,
        output=smpl_video_path,
        progress_bar=progress.tqdm,
    )

    return smpl_video_path


theme = gr.themes.Default(
    primary_hue="blue", secondary_hue="gray", text_size=sizes.text_lg
)


css = """
h1 {
    text-align: center;
    display:block;
}

h2 {
    text-align: center;
    display:block;
}

.centered {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown(WEBSITE)

    with gr.Row():
        btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            cancel = gr.Button("Cancel", variant="stop")
            clear = gr.Button("Clear", variant="secondary")

    gr.Markdown("<h3> Set of parameters </h3>")
    with gr.Row():
        seed = gr.Number(
            value=0,
            label="Seed",
            info="Deterministic outputs for reproducibility",
            scale=1,
        )
        seconds = gr.Number(
            value=DEFAULT_MAX_S,
            label="Duration in seconds",
            info="For the total timeline",
        )

        gender = gr.Radio(
            ["male", "female", "neutral"],
            value="male",
            label="SMPL body model",
        )

        overlap_s = gr.Number(
            value=1.0,
            minimum=0.0,
            maximum=DEFAULT_MAX_S,
            label="Overlap in seconds",
        )

        guidance_weight = gr.Number(
            value=2.5,
            minimum=1.0,
            maximum=DEFAULT_MAX_S,
            label="Guidance scale",
        )

        with gr.Column():
            number_of_texts = gr.Number(
                value=DEFAULT_NTEXT,
                maximum=MAX_TEXT,
                label="Number of texts",
                precision=0,
            )
            with gr.Row():
                adding_text = gr.Button("Add")
                removing_text = gr.Button("Remove")

    begins = []
    ends = []
    texts = []
    body_parts = []
    for i in range(MAX_TEXT):
        visible = False
        if i < DEFAULT_NTEXT:
            visible = True

        with gr.Row():
            with gr.Column(scale=1):
                texts.append(
                    gr.Textbox(
                        placeholder="Type the motion you want to generate",
                        show_label=True,
                        label="Text prompt",
                        value=DEFAULT_TEXTS[i],
                        visible=visible,
                    )
                )
                body_parts.append(
                    gr.CheckboxGroup(
                        choices=["left arm", "right arm", "legs", "head", "spine"],
                        label="Body parts",
                        visible=visible,
                        value=DEFAULT_BODY_PARTS[i],
                    )
                )
            with gr.Column(scale=1):
                begins.append(
                    gr.Slider(
                        0,
                        DEFAULT_MAX_S,
                        value=DEFAULT_STARTS[i],
                        visible=visible,
                        label="Start",
                    )
                )
                ends.append(
                    gr.Slider(
                        0,
                        DEFAULT_MAX_S,
                        value=DEFAULT_ENDS[i],
                        visible=visible,
                        label="End",
                    )
                )

    inputs = (
        [
            seconds,
            number_of_texts,
            guidance_weight,
            overlap_s,
            seed,
        ]
        + begins
        + ends
        + texts
        + body_parts
    )

    init_plot = plot_box(
        DEFAULT_MAX_S,
        DEFAULT_NTEXT,
        DEFAULT_STARTS + DEFAULT_ENDS + DEFAULT_TEXTS + DEFAULT_BODY_PARTS,
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<h3> Visualization of the inputs </h3>")
            plot = gr.Plot(label="Bounding box of texts", value=init_plot)
        with gr.Column():
            gr.Markdown("<h3> Outputs </h3>")
            smplrifke_outputs = gr.File(label="Features (smplrifke)")
            joints_outputs = gr.File(label="Joints")
            vertices_outputs = gr.File(label="Vertices")
            smpl_outputs = gr.File(label="SMPL parameters")

    with gr.Row():
        with gr.Column():
            gr.Markdown("<h3> Joints rendering </h3>")
            video_j = gr.Video(autoplay=True)
        with gr.Column():
            gr.Markdown("<h3> SMPL rendering </h3>")
            video_s = gr.Video(autoplay=True)

    def adding_text_fn(n):
        return min(n + 1, MAX_TEXT)

    def removing_text_fn(n):
        return max(0, n - 1)

    # hidden
    smplrifke_path = gr.Textbox(
        placeholder="",
        show_label=False,
        value="",
        visible=False,
    )
    vertices_path = gr.Textbox(
        placeholder="",
        show_label=False,
        value="",
        visible=False,
    )

    adding_text.click(
        fn=adding_text_fn, inputs=number_of_texts, outputs=number_of_texts
    )
    removing_text.click(
        fn=removing_text_fn, inputs=number_of_texts, outputs=number_of_texts
    )

    plot_box_inputs = [seconds, number_of_texts] + begins + ends + texts + body_parts

    btn_plot_event = btn.click(
        fn=plot_box,
        inputs=plot_box_inputs,
        outputs=plot,
    )
    btn_gen_event = btn_plot_event.then(
        fn=generate,
        inputs=inputs,
        outputs=[video_j, smplrifke_path],
    )
    btn_get_smpl_data = btn_gen_event.then(
        fn=get_data,
        inputs=[smplrifke_path, gender],
        outputs=[
            vertices_path,
            smplrifke_outputs,
            joints_outputs,
            vertices_outputs,
            smpl_outputs,
        ],
    )
    btn_render_smpl = btn_get_smpl_data.then(
        fn=render_smpl,
        inputs=vertices_path,
        outputs=video_s,
    )

    btn_events = [btn_plot_event, btn_gen_event, btn_get_smpl_data, btn_render_smpl]

    def cancel_process():
        return [init_plot, None, None, None, None, None, None]

    output_to_clear = [
        plot,
        video_j,
        video_s,
        smplrifke_outputs,
        joints_outputs,
        vertices_outputs,
        smpl_outputs,
    ]
    clear.click(fn=cancel_process, outputs=output_to_clear)
    cancel.click(fn=cancel_process, outputs=output_to_clear, cancels=btn_events)

    number_of_texts.change(
        fn=update_numbers,
        inputs=[number_of_texts, seconds] + begins + ends,
        outputs=begins + ends + texts + body_parts,
    ).then(
        fn=plot_box,
        inputs=plot_box_inputs,
        outputs=plot,
    )
    seconds.submit(
        fn=update_numbers,
        inputs=[number_of_texts, seconds] + begins + ends,
        outputs=begins + ends + texts + body_parts,
    ).then(
        fn=plot_box,
        inputs=plot_box_inputs,
        outputs=plot,
    )

    for val in texts:
        val.submit(
            fn=plot_box,
            inputs=plot_box_inputs,
            outputs=plot,
        )
        val.blur(
            fn=plot_box,
            inputs=plot_box_inputs,
            outputs=plot,
        )

    for val in begins + ends:
        val.release(
            fn=plot_box,
            inputs=plot_box_inputs,
            outputs=plot,
        )


demo.queue()
demo.launch(share=SHARE)
