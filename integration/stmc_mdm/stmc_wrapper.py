# Hack to add STMC functionality without copying everything again
# modified from motion-diffusion-model/diffusion/gaussian_diffusion.py
# and model/mdm.py
#
# Must be in in sync with the original files
# I only added the part enclosed by the ###

import types
import torch
import torch as th
import numpy as np

from diffusion.gaussian_diffusion import (
    ModelMeanType,
    ModelVarType,
    _extract_into_tensor,
)

from stmc_mdm.stmc import combine_features_intervals


# Monkey patch
def STMC_wrapper(model, diffusion):
    model.forward = types.MethodType(__mdm_forward, model)
    diffusion.p_mean_variance = types.MethodType(__p_mean_variance, diffusion)


def __mdm_forward(self, x, timesteps, y=None):
    """
    x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
    timesteps: [batch_size] (int)
    """

    bs, njoints, nfeats, nframes = x.shape
    emb = self.embed_timestep(timesteps)  # [1, bs, d]

    force_mask = y.get("uncond", False)
    #############################################################
    # BEGIN
    ############################################################
    if "text" in self.cond_mode:
        # "" considered as unconditional
        mask_none = [x == "" for x in y["text"]]
        enc_text = self.encode_text(y["text"])
        enc_text[mask_none] = 0 * enc_text[mask_none]
        # make it zero for uncond
        emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
    #############################################################
    # END
    ############################################################

    if "action" in self.cond_mode:
        action_emb = self.embed_action(y["action"])
        emb += self.mask_cond(action_emb, force_mask=force_mask)

    if self.arch == "gru":
        x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
        emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
        emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
        emb_gru = emb_gru.reshape(
            bs, self.latent_dim, 1, nframes
        )  # [bs, d, 1, #frames]
        x = torch.cat((x_reshaped, emb_gru), axis=1)  # [bs, d+joints*feat, 1, #frames]

    x = self.input_process(x)

    if self.arch == "trans_enc":
        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[
            1:
        ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

    elif self.arch == "trans_dec":
        if self.emb_trans_dec:
            xseq = torch.cat((emb, x), axis=0)
        else:
            xseq = x
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        if self.emb_trans_dec:
            output = self.seqTransDecoder(tgt=xseq, memory=emb)[
                1:
            ]  # [seqlen, bs, d] # FIXME - maybe add a causal mask
        else:
            output = self.seqTransDecoder(tgt=xseq, memory=emb)
    elif self.arch == "gru":
        xseq = x
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
        output, _ = self.gru(xseq)

    output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
    return output


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

    ############################################################
    # BEGIN
    ############################################################
    if "stmc_infos" in model_kwargs["y"]:
        infos = model_kwargs["y"]["stmc_infos"]
        # split the vector
        all_intervals = infos["all_intervals"]
        max_frames = model_kwargs["y"]["max_frames"]

        # MDM takes fix size input
        # so "start + max_frames" instead of "end"
        x_lst = []
        for idx, intervals in enumerate(all_intervals):
            x_lst.extend(
                [x[idx, :, :, y.start : y.start + max_frames] for y in intervals]
            )

        # can stack without padding as they are all the same size
        # because MDM take fix size input
        new_x = th.stack(x_lst, axis=0)

        # recreate individual kwargs
        new_model_kwargs = {}
        new_model_kwargs["y"] = {
            x: y for x, y in model_kwargs["y"].items() if x != "stmc_infos"
        }
        # flatten version of texts for MDM forward pass
        texts = infos["all_texts"]
        new_model_kwargs["y"]["text"] = texts

        n_elements = len(texts)

        assert all(t == t[0])
        new_t = t[0].repeat(n_elements)

        scales = model_kwargs["y"]["scale"]
        assert all(scales == scales[0])
        new_scales = scales[0].repeat(n_elements)
        new_model_kwargs["y"]["scale"] = new_scales

        # classifier-free guidance
        # already included in the model
        new_model_output = model(
            new_x, self._scale_timesteps(new_t), **new_model_kwargs
        )
        # make new x for
        model_output = 0 * x
        # inplace
        combine_features_intervals(new_model_output, infos, model_output)

    else:
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
    ############################################################
    # END
    ############################################################

    if (
        "inpainting_mask" in model_kwargs["y"].keys()
        and "inpainted_motion" in model_kwargs["y"].keys()
    ):
        inpainting_mask, inpainted_motion = (
            model_kwargs["y"]["inpainting_mask"],
            model_kwargs["y"]["inpainted_motion"],
        )
        assert (
            self.model_mean_type == ModelMeanType.START_X
        ), "This feature supports only X_start pred for mow!"
        assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
        model_output = (model_output * ~inpainting_mask) + (
            inpainted_motion * inpainting_mask
        )
        # print('model_output', model_output.shape, model_output)
        # print('inpainting_mask', inpainting_mask.shape, inpainting_mask[0,0,0,:])
        # print('inpainted_motion', inpainted_motion.shape, inpainted_motion)

    if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)
        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
    else:
        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        # print('model_variance', model_variance)
        # print('model_log_variance',model_log_variance)
        # print('self.posterior_variance', self.posterior_variance)
        # print('self.posterior_log_variance_clipped', self.posterior_log_variance_clipped)
        # print('self.model_var_type', self.model_var_type)

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

    def process_xstart(x):
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            # print('clip_denoised', clip_denoised)
            return x.clamp(-1, 1)
        return x

    if self.model_mean_type == ModelMeanType.PREVIOUS_X:
        pred_xstart = process_xstart(
            self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
        )
        model_mean = model_output
    elif self.model_mean_type in [
        ModelMeanType.START_X,
        ModelMeanType.EPSILON,
    ]:  # THIS IS US!
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
    else:
        raise NotImplementedError(self.model_mean_type)

    assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
    }
