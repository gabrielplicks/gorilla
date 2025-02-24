import torch

from typing import List, Union



def process_batch(batch: List[List[Union[int, float]]], shape: List[int]) -> List[torch.Tensor]:

    """Processes a batch of multi-dimensional arrays into tensors.

    Args:

        batch: A list of multi-dimensional arrays.

        shape: The shape of the tensor to be created.

    """

    processed_tensors = []

    for array in batch:

        tensor = torch.Tensor(array)

        tensor = tensor.reshape(shape)

        processed_tensors.append(tensor)

    return processed_tensors

