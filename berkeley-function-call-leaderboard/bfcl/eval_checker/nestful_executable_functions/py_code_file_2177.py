import torch



def threshold_tensor(tensor: torch.Tensor, threshold: float) -> torch.Tensor:

    """

    Converts a multi-dimensional tensor to a binary tensor based on a threshold.

    The value of each element in the output tensor is 1 if it is greater than or equal to the threshold, and 0 otherwise.



    Args:

        tensor: The input multi-dimensional tensor.

        threshold: The scalar threshold.

    """

    comparison = tensor.ge(threshold)

    binary_tensor = comparison.int()

    return binary_tensor

