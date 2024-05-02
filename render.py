import os
import logging
import hydra

from hydra.utils import instantiate
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def T(x):
    import torch
    import numpy as np

    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))


@hydra.main(config_path="configs", config_name="render", version_base="1.3")
def render(c: DictConfig):
    logger.info("Rendering script")
    import numpy as np

    motions = np.load(c.path)

    if motions.shape[1] == 6890:
        renderer = instantiate(c.smpl_renderer)
    else:
        renderer = instantiate(c.joints_renderer)

    ext = "." + c.ext.replace(".", "")
    if c.out_path is None:
        c.out_path = os.path.splitext(c.path)[0] + ext

    logger.info(f"The video will be renderer there: {c.out_path}")

    if len(motions) == 1:
        motions = motions[0]

    if c.y_is_z_axis:
        x, mz, my = T(motions)
        motions = T(np.stack((x, -my, mz), axis=0))

    renderer(motions, c.out_path, fps=c.fps)


if __name__ == "__main__":
    render()
