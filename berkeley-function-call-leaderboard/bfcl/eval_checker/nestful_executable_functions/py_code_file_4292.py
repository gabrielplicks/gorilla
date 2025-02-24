import torch.nn as nn



def calculate_model_size(model: nn.Module) -> int:

    """Calculates the total size of a neural network's parameters in bytes.



    Args:

        model: The PyTorch model.



    Returns:

        The total size of the model's parameters in bytes.

    """

    total_size = 0

    for param in model.parameters():

        param_size = param.numel() * param.element_size()

        total_size += param_size



    return total_size

