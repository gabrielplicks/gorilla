from typing import Union

import torch



def count_trainable_parameters(model: torch.nn.Module) -> int:

    """Counts the number of trainable parameters in a PyTorch model.



    Args:

        model: The PyTorch model.



    Returns:

        The number of trainable parameters in the model.

    """

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params

