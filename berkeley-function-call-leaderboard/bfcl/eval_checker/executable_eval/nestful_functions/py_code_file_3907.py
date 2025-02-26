import torch



def clamp_and_convert_to_uint8(x: torch.Tensor) -> torch.Tensor:

    """Clamps the values in a PyTorch tensor to the range [0, 1] and converts them to the uint8 dtype.



    Args:

        x: The input PyTorch tensor representing an image with dimensions (C, H, W).



    Returns:

        The clamped and converted tensor.

    """

    return x.clamp(0, 1).byte()

