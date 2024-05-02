import logging
import torch
import warnings

logger = logging.getLogger("torch.distributed.nn.jit.instantiator")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("OpenGL.acceleratesupport")
logger.setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore", "The PyTorch API of nested tensors is in prototype stage*"
)

warnings.filterwarnings("ignore", "Converting mask without torch.bool dtype to bool")
warnings.filterwarnings("ignore", "enable_nested_tensor is True")

torch.set_float32_matmul_precision("high")
