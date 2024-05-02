import torch


def add_constants(model, betas, timesteps):
    assert (betas >= 0).all() and (betas <= 1).all()

    # buffers -> move in the GPU with model.to() auto
    # persistent=False -> not saved in state_dict

    model.register_buffer("betas", betas, persistent=False)

    model.register_buffer("alphas", 1.0 - model.betas, persistent=False)
    model.register_buffer(
        "alphas_cumprod",
        torch.cumprod(model.alphas, axis=0),
        persistent=False
    )
    model.register_buffer(
        "one_minus_alphas_cumprod",
        1.0 - model.alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "inv_one_minus_alphas_cumprod",
        1.0 / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "alphas_cumprod_prev",
        torch.cat([torch.ones(1), model.alphas_cumprod[:-1]]),
        persistent=False
    )
    model.register_buffer(
        "alphas_cumprod_next",
        torch.cat([model.alphas_cumprod[:-1], torch.ones(0)]),
        persistent=False
    )
    # calculations for diffusion q(x_t | x_{t-1}) and others
    model.register_buffer(
        "sqrt_alphas_cumprod",
        torch.sqrt(model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_one_minus_alphas_cumprod",
        torch.sqrt(1.0 - model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_one_minus_alphas_cumprod_prev",
        torch.sqrt(1 - model.alphas_cumprod_prev),
        persistent=False
    )
    model.register_buffer(
        "inv_sqrt_alphas_cumprod",
        1.0 / model.sqrt_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "sqrt_inv_alphas_cumprod_minus_one",
        torch.sqrt(1.0 / model.alphas_cumprod - 1),
        persistent=False
    )
    model.register_buffer(
        "sqrt_alphas_cumprod_over_one_minus_alphas_cumprod",
        torch.sqrt(model.alphas_cumprod) / (1.0 - model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_alphas_cumprod_over_sqrt_one_minus_alphas_cumprod",
        torch.sqrt(model.alphas_cumprod / (1.0 - model.alphas_cumprod)),
        persistent=False
    )
    model.register_buffer(
        "one_minus_alphas_cumprod_over_sqrt_alphas_cumprod",
        (1.0 - model.alphas_cumprod) / torch.sqrt(model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_inv_one_minus_alphas_cumprod",
        torch.sqrt(1.0 / (1.0 - model.alphas_cumprod)),
        persistent=False
    )
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    model.register_buffer(
        "posterior_variance",
        model.betas * (1.0 - model.alphas_cumprod_prev) / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_coef1",
        model.betas * torch.sqrt(model.alphas_cumprod_prev) / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_coef2",
        (1.0 - model.alphas_cumprod_prev) * torch.sqrt(model.alphas) / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_eps_coef1",
        1.0 / torch.sqrt(model.alphas),
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_eps_coef2",
        betas / (model.sqrt_one_minus_alphas_cumprod * torch.sqrt(model.alphas)),
        persistent=False
    )
    # same coef for xt
    model.register_buffer(
        "posterior_mean_score_coef1",
        model.posterior_mean_eps_coef1,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_score_coef2",
        betas / torch.sqrt(model.alphas),
        persistent=False
    )
