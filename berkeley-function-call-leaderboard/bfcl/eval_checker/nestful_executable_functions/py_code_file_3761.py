import torch



def process_tensor(x: torch.Tensor) -> torch.Tensor:

    """Processes a tensor and returns a boolean tensor with the same shape.



    The returned tensor has `True` elements where the input tensor has at least three elements that are `True`.

    If the input tensor is `None`, then the output tensor is also `None`.



    Args:

        x: The input tensor.



    Returns:

        A boolean tensor with the same shape as the input tensor.

    """

    if x is None:

        return None



    num_true = x.sum(dim=1)

    y = torch.zeros_like(x, dtype=torch.bool)

    y[num_true >= 3] = True

    return y

