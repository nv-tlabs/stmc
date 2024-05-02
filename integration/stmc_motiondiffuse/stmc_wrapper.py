# Hack to add STMC functionality without copying everything again
# modified from text2motion/models/gaussian_diffusion.py
# and text2motion/trainers/ddpm_trainer.py
# Must be in in sync with the original files
# For gaussian diffusion, I only added the part enclosed by the ###

import types
import torch
import torch as th
import numpy as np

from models.gaussian_diffusion import _extract_into_tensor
from stmc_motiondiffuse.stmc import combine_features_intervals


# Monkey patch
def STMC_wrapper(trainer, baseline):
    if "sinc" in baseline:
        trainer.generate_batch = types.MethodType(__generate_batch_sinc, trainer)
    else:
        trainer.generate_batch = types.MethodType(__generate_batch, trainer)
        trainer.diffusion.p_mean_variance = types.MethodType(
            __p_mean_variance, trainer.diffusion
        )


def __generate_batch(self, caption, m_lens, dim_pose, infos):
    xf_proj, xf_out = self.encoder.encode_text(caption, self.device)

    # B = len(caption)
    B = len(m_lens)
    # T = min(m_lens.max(), self.encoder.num_frames)
    T = m_lens.max() + self.encoder.num_frames
    # make sure to have room
    assert max(infos["all_lengths"]) <= self.encoder.num_frames

    output = self.diffusion.p_sample_loop(
        self.encoder,
        (B, T, dim_pose),
        clip_denoised=False,
        progress=True,
        model_kwargs={
            "xf_proj": xf_proj,
            "xf_out": xf_out,
            "length": infos["all_lengths"],
            "stmc_infos": infos,
        },
    )
    return output


def __generate_batch_sinc(self, caption, m_lens, dim_pose, infos):
    xf_proj, xf_out = self.encoder.encode_text(caption, self.device)

    B = len(caption)
    T = self.encoder.num_frames
    output = self.diffusion.p_sample_loop(
        self.encoder,
        (B, T, dim_pose),
        clip_denoised=False,
        progress=True,
        model_kwargs={"xf_proj": xf_proj, "xf_out": xf_out, "length": m_lens},
    )
    return output


# add stmc functionality
def __p_mean_variance(
    self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
):
    """
    Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
    the initial x, x_0.

    :param model: the model, which takes a signal and a batch of timesteps
                  as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :param clip_denoised: if True, clip the denoised signal into [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample. Applies before
        clip_denoised.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict with the following keys:
             - 'mean': the model mean output.
             - 'variance': the model variance output.
             - 'log_variance': the log of 'variance'.
             - 'pred_xstart': the prediction for x_0.
    """
    if model_kwargs is None:
        model_kwargs = {}

    B, C = x.shape[:2]
    assert t.shape == (B,)

    # model_kwargs.keys()
    # dict_keys(['xf_proj', 'xf_out', 'length'])
    # x.shape
    # torch.Size([1, 120, 263])

    ############################################################
    # BEGIN
    ############################################################
    if "stmc_infos" in model_kwargs:
        infos = model_kwargs["stmc_infos"]
        # split the vector
        all_intervals = infos["all_intervals"]
        max_frames = infos["max_frames"]

        x_lst = []
        for idx, intervals in enumerate(all_intervals):
            # all have the size max_frames
            # faster to implement than collate
            # but equivalent here since MotionDiffuse use masking
            x_lst.extend([x[idx, y.start : y.start + max_frames] for y in intervals])
        new_x = th.stack(x_lst, axis=0)

        new_model_kwargs = {x: y for x, y in model_kwargs.items() if x != "stmc_infos"}

        n_elements = len(new_x)

        assert all(t == t[0])
        new_t = t[0].repeat(n_elements)

        new_model_output = model(
            new_x, self._scale_timesteps(new_t), **new_model_kwargs
        )

        new_pred_xstart = self._predict_xstart_from_eps(
            x_t=new_x, t=new_t, eps=new_model_output
        )
        # make new x for
        pred_xstart = 0 * x
        # inplace
        combine_features_intervals(new_pred_xstart, infos, pred_xstart)
    else:
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
    ############################################################
    # END
    ############################################################

    model_variance, model_log_variance = (
        self.posterior_variance,
        self.posterior_log_variance_clipped,
    )
    model_variance = _extract_into_tensor(model_variance, t, x.shape)
    model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

    model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

    assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
    }
