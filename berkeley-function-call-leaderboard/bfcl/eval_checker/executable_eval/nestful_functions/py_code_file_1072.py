import torch

from typing import Tuple



def count_non_zero_pixels(tensor1: torch.Tensor, tensor2: torch.Tensor) -> int:

    """Counts the number of elements where tensor1 and tensor2 are equal and non-zero.



    Args:

        tensor1: A tensor.

        tensor2: A tensor of the same shape as tensor1.



    Raises:

        ValueError: If the tensors are not of the same shape.

    """

    if tensor1.shape != tensor2.shape:

        raise ValueError("Tensors must be of the same shape.")

    count = 0

    for i in range(tensor1.shape[0]):

        for j in range(tensor1.shape[1]):

            if tensor1[i, j] == tensor2[i, j] and tensor1[i, j] != 0:

                count += 1



    return count

