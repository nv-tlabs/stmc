import os
import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        super().__init__()
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        mean = torch.load(self.mean_path)
        std = torch.load(self.std_path)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x
