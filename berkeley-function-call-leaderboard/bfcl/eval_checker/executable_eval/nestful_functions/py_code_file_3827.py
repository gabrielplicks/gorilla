import torch

from typing import Tuple



def resize_and_normalize(image: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:

    """Resizes an image to a target shape and normalizes the pixel values to the range [0, 1].



    Args:

        image: A PyTorch tensor representing the image.

        shape: A tuple of two integers representing the desired height and width of the resized image.



    Returns:

        A PyTorch tensor of the resized and normalized image.

    """

    resized_image = torch.nn.functional.interpolate(image, size=shape, mode='nearest')

    normalized_image = torch.clamp((resized_image - torch.min(resized_image)) / (torch.max(resized_image) - torch.min(resized_image)), min=0, max=1)



    return normalized_image

