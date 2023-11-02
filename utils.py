import logging
import numpy as np
import torch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _get_value(labels):
    """
    get the value based on the pointer

    Args:
        labels: a tensor of length 4. where labels[0] is the pointer

    Returns:
        The value
    """

    pointer = labels[0]
    if 0 <= pointer <= 3:
        value = labels[1]
    elif 4 <= pointer <= 6:
        value = labels[2]
    else:
        value = labels[3]

    return value
