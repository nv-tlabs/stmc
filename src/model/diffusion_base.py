import torch
import einops

from pytorch_lightning import LightningModule
from .constants import add_constants


def extract(coef, t, tensor):
    shape = tensor.shape
    return coef[t].reshape(shape[0], *((1,) * (len(shape) - 1))).to(tensor)


# Base class for having distribution constants and methods
# Equation taken from this great pdf:
# https://arxiv.org/pdf/2208.11970.pdf
class DiffuserBase(LightningModule):
    def __init__(self, schedule, timesteps):
        super().__init__()

        self.linear_scale = 1.0
        betas = schedule(timesteps)
        # add all the constants (betas / alphas / alpha_cumprod etc)
        add_constants(self, betas, timesteps)

    def on_save_checkpoint(self, checkpoint):
        if False:
            # remove tmr checkpoints
            checkpoint["state_dict"] = {
                x: y
                for x, y in checkpoint["state_dict"].items()
                if not x.startswith("tmr")
            }

    def sample_from_distribution_function(self, function, *args, noise=None):
        if noise is None:
            noise = torch.randn_like(args[0])

        mean, sigma = function(*args)
        sample = mean + sigma * noise
        return sample

    # Equation 115 (page 15)
    def eps_to_xstart(self, eps, xt, t):
        xstart = (
            extract(self.inv_sqrt_alphas_cumprod, t, xt) * xt
            - extract(self.sqrt_inv_alphas_cumprod_minus_one, t, xt) * eps
        )
        return xstart

    # Equation 133 (page 16)
    def score_to_xstart(self, score, xt, t):
        xstart = (
            extract(self.inv_sqrt_alphas_cumprod, t, xt) * xt
            + extract(self.one_minus_alphas_cumprod_over_sqrt_alphas_cumprod, t, xt)
            * score
        )
        return xstart

    # Equation 151 (page 17)
    def eps_to_score(self, eps, xt, t):
        score = -extract(self.sqrt_inv_one_minus_alphas_cumprod, t, eps) * eps
        return score

    # Equation 69 (page 11) + equation 151 (page 17)
    def xstart_to_score(self, xstart, xt, t):
        score = (
            extract(self.sqrt_alphas_cumprod_over_one_minus_alphas_cumprod, t, xt)
            * xstart
            - extract(self.inv_one_minus_alphas_cumprod, t, xt) * xt
        )
        return score

    # Equation 69 (page 11)
    def xstart_to_eps(self, xstart, xt, t):
        eps = (
            extract(self.sqrt_inv_one_minus_alphas_cumprod, t, xt) * xt
            - extract(
                self.sqrt_alphas_cumprod_over_sqrt_one_minus_alphas_cumprod, t, xt
            )
            * xstart
        )
        return eps

    # Equation 151 (page 17)
    def score_to_eps(self, score, xt, t):
        eps = -extract(self.sqrt_one_minus_alphas_cumprod, t, score) * score
        return eps

    # Equation 70 (page 11)
    def q_distribution(self, xstart, t):
        """
        q(xt | xstart) -> forward diffusion process
        """
        mean = self.linear_scale * extract(self.sqrt_alphas_cumprod, t, xstart) * xstart
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, xstart)
        return mean, sigma

    def q_sample(self, xstart, t, noise=None):
        """
        Sample from q(xt | xstart) -> diffuse the data
        """
        sample = self.sample_from_distribution_function(
            self.q_distribution, xstart, t, noise=noise
        )
        return sample

    # Equation 84 (page 12)
    def q_posterior_distribution_from_xstart_and_xt(self, xstart, xt, t):
        """
        q(xt-1 | xstart, xt) -> Exact denoising distribution (when xstart is known)
        """
        mean = (
            extract(self.posterior_mean_coef1, t, xt) * xstart
            + extract(self.posterior_mean_coef2, t, xt) * xt
        )
        sigma = extract(self.posterior_variance, t, xt)
        return mean, sigma

    def q_posterior_sample_from_xstart_and_xt(self, xstart, xt, t, noise=None):
        """
        Sample from q(xt-1 | xstart, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_xstart_and_xt, xstart, xt, t, noise=noise
        )
        return sample

    # Equation 84 (page 12) + equation 69 (page 11)
    def q_posterior_distribution_from_eps_and_xt(self, eps, xt, t):
        """
        q(xt-1 | eps, xt) -> Exact denoising distribution (when eps is known)
        """
        mean = (
            extract(self.posterior_mean_eps_coef1, t, xt) * xt
            - extract(self.posterior_mean_eps_coef2, t, xt) * eps
        )

        sigma = extract(self.posterior_variance, t, xt)
        return mean, sigma

    def q_posterior_sample_from_eps_and_xt(self, eps, xt, t, noise=None):
        """
        Sample from q(xt-1 | eps, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_eps_and_xt, eps, xt, t, noise=noise
        )
        return sample

    # Equation 84 (page 12) + equation 69 (page 11)
    def q_posterior_distribution_from_score_and_xt(self, score, xt, t):
        """
        q(xt-1 | score, xt) -> Exact denoising distribution (when score is known)
        """
        mean = (
            extract(self.posterior_mean_score_coef1, t, xt) * xt
            + extract(self.posterior_mean_score_coef2, t, xt) * score
        )

        sigma = extract(self.posterior_variance, t, xt)
        return mean, sigma

    def q_posterior_sample_from_score_and_xt(self, score, xt, t, noise=None):
        """
        Sample from q(xt-1 | score, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_score_and_xt, score, xt, t, noise=noise
        )
        return sample

    def q_posterior_distribution_from_output_and_xt(self, output, xt, t):
        if self.prediction == "x":
            return self.q_posterior_distribution_from_xstart_and_xt(output, xt, t)
        elif self.prediction == "eps":
            return self.q_posterior_distribution_from_eps_and_xt(output, xt, t)
        elif self.prediction == "score":
            return self.q_posterior_distribution_from_score_and_xt(output, xt, t)
        else:
            raise NotImplementedError

    def q_posterior_sample_from_output_and_xt(self, output, xt, t, noise=None):
        """
        Sample from q(xt-1 | output, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_output_and_xt, output, xt, t, noise=noise
        )
        return sample

    def _format_str(self, string):
        assert string in ["output", "x", "xstart", "eps", "score"]
        if string == "output":
            string = self.prediction
        if string == "x":
            string = "xstart"
        return string

    def to_space(self, space_in, space_out, val, xt, t):
        space_in = self._format_str(space_in)
        space_out = self._format_str(space_out)
        if space_in == space_out:
            return val
        return getattr(self, f"{space_in}_to_{space_out}")(val, xt, t)

    def to_xstart(self, string, val, xt, t):
        return self.to_space(string, "xstart", val, xt, t)

    def to_eps(self, string, val, xt, t):
        return self.to_space(string, "eps", val, xt, t)

    def to_score(self, string, val, xt, t):
        return self.to_space(string, "score", val, xt, t)

    def to_output(self, string, val, xt, t):
        return self.to_space(string, "output", val, xt, t)

    def output_to(self, string, val, xt, t):
        return self.to_space("output", string, val, xt, t)

    def q_sample_one_x_all_t(self, x):
        t = torch.arange(0, self.timesteps, device=x.device)
        # max time steps is T-1
        # t=0 means the first diffusion step q_sample(x0, t=0) -> x1
        # so the last one is q_sample(xT-1, t=T-1) -> xT
        # there is T-1 diffusion output
        xstart = einops.repeat(x, "time feats -> diff time feats", diff=self.timesteps)
        xt = self.q_sample(xstart=xstart, t=t)
        return xt

    @torch.no_grad()
    def p_distribution(self, xt, y, t, indices, guid_params=None, return_output=False):
        output = self.guided_forward(xt, y, t, indices, guid_params=guid_params)
        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)

        if return_output:
            return mean, sigma, output
        return mean, sigma


if __name__ == "__main__":
    # Test consistency
    # from .schedule.linear import LinearBetaSchedule
    from .schedule.cosine import CosineBetaSchedule

    schedule = CosineBetaSchedule()
    timesteps = 1000
    diffusion = DiffuserBase(schedule, timesteps)

    shape = (10, 60, 73)
    xstart = torch.ones(shape) + 0.1 * torch.randn(shape)
    xstart = xstart.to(torch.float64)

    t = torch.full((10,), timesteps - 200)
    xt = diffusion.q_sample(xstart, t)

    eps_x = diffusion.xstart_to_eps(xstart, xt, t)
    score_x = diffusion.xstart_to_score(xstart, xt, t)

    xstart_eps = diffusion.eps_to_xstart(eps_x, xt, t)
    score_eps = diffusion.eps_to_score(eps_x, xt, t)

    xstart_s = diffusion.score_to_xstart(score_x, xt, t)
    eps_s = diffusion.score_to_eps(score_x, xt, t)

    # check consistency
    assert torch.linalg.norm(xstart - xstart_eps) < 1e-10
    assert torch.linalg.norm(xstart - xstart_s) < 1e-10
    assert torch.linalg.norm(eps_x - eps_s) < 1e-10

    mean_x, sigma_x = diffusion.q_posterior_distribution_from_xstart_and_xt(
        xstart, xt, t
    )
    mean_eps, sigma_eps = diffusion.q_posterior_distribution_from_eps_and_xt(
        eps_x, xt, t
    )
    mean_s, sigma_s = diffusion.q_posterior_distribution_from_score_and_xt(
        score_x, xt, t
    )

    assert torch.linalg.norm(mean_x - mean_s) < 1e-10
    assert torch.linalg.norm(mean_x - mean_eps) < 1e-10
    assert torch.linalg.norm(mean_s - mean_eps) < 1e-10

    # Should be the same
    assert (sigma_x == sigma_eps).all()
    assert (sigma_x == sigma_s).all()
    assert (sigma_eps == sigma_s).all()

    print("The transformations are consistent.")
