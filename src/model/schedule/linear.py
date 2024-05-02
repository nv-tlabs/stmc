import torch


class LinearBetaSchedule:
    def __init__(self, start=0.0001, end=0.02):
        self.start = start
        self.end = end

    def __call__(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * self.start
        beta_end = scale * self.end
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
