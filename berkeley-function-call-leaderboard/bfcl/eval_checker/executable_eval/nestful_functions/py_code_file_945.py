from typing import Tuple



def calculate_image_size(width: int, height: int, resolution: int) -> Tuple[int, int]:

    """Calculates the size of an image in bytes.



    Args:

        width: The width of the image in pixels.

        height: The height of the image in pixels.

        resolution: The resolution of the image (32-bit or 64-bit).



    Returns:

        A tuple of the width and height of the image in bytes.

    """

    num_pixels = width * height

    num_bytes_per_pixel = 4 if resolution == 32 else 8

    size_in_bytes = num_pixels * num_bytes_per_pixel

    return size_in_bytes

