from typing import Any, Dict, List, Union



import torch

from torch.optim import Optimizer



def get_lr(optimizer: Optimizer) -> float:

    """Retrieves the learning rate of the first parameter group.



    Args:

        optimizer: The PyTorch optimizer.



    Returns:

        The learning rate of the first parameter group.

    """

    for param_group in optimizer.param_groups:

        return param_group['lr']

