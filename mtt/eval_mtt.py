import src.prepare  # noqa

from .stmc import read_timelines

from .metrics import calculate_activation_statistics_normalized
from .load_tmr_model import load_tmr_model_easy
from .load_mtt_texts_motions import load_mtt_texts_motions
from .stmc_metrics import get_gt_metrics, get_exp_metrics, print_latex

from .experiments import experiments

device = "cpu"

# FOLDER to evaluate
fps = 20.0
mtt_timelines = "mtt/MTT.txt"

tmr_forward = load_tmr_model_easy(device)
texts_gt, motions_guofeats_gt = load_mtt_texts_motions(fps)

text_dico = {t: i for i, t in enumerate(texts_gt)}

text_latents_gt = tmr_forward(texts_gt)
motion_latents_gt = tmr_forward(motions_guofeats_gt)

metric_names = [32 * " ", "R1  ", "R3  ", "M2T S", "M2M S", "FID+ ", "Trans"]
print(" & ".join(metric_names))

gt_metrics = get_gt_metrics(motion_latents_gt, text_latents_gt, motions_guofeats_gt)
print_latex("GT", gt_metrics)

gt_mu, gt_cov = calculate_activation_statistics_normalized(motion_latents_gt.numpy())

timelines = read_timelines(mtt_timelines, fps)
assert len(timelines) == 500
timelines_dict = {str(idx).zfill(4): timeline for idx, timeline in enumerate(timelines)}

for exp in experiments:
    if exp.get("skip", False):
        continue
    metrics = get_exp_metrics(
        exp,
        tmr_forward,
        text_dico,
        timelines_dict,
        gt_mu,
        gt_cov,
        text_latents_gt,
        motion_latents_gt,
        fps,
    )
    print_latex(exp["name"], metrics)
