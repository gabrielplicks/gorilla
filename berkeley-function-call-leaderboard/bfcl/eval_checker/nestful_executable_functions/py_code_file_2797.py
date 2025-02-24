import torch

import torch.nn as nn



def gram_matrix(input_tensor: torch.Tensor, gram_matrices: list[nn.Module]) -> list[torch.Tensor]:

    """Computes a gram matrix for each convolutional layer in the given list.



    Args:

        input_tensor: A tensor with dimensions (batch_size, num_channels, height, width).

        gram_matrices: A list of PyTorch modules that represent the convolutional layers.



    Returns:

        A list of the computed gram matrices.

    """

    result = []

    for gram_matrix in gram_matrices:

        gram = torch.mm(input_tensor.view(input_tensor.size(1), -1),

                        input_tensor.view(input_tensor.size(1), -1).t())

        result.append(gram)

    return result

