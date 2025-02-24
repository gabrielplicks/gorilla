import torch



def sum_batch_channel(x: torch.Tensor) -> torch.Tensor:

    """Computes the sum of the values in a PyTorch tensor along the batch and channel dimensions.



    Args:

        x: A PyTorch tensor with shape `(N, C, H, W)`, where N is the batch size, C is the number of channels, H is the height, and W is the width.



    Returns:

        A PyTorch tensor with shape `(H, W)` containing the sum of the values in the input tensor along the batch and channel dimensions.

    """

    return x.sum(dim=(0, 1))

