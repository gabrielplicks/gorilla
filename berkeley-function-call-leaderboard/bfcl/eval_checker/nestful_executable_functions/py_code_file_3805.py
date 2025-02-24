from typing import Tuple



def calculate_resolution(width: int, height: int, pixels: int) -> str:

    """Calculates the resolution of an image given its dimensions and number of pixels.

    Args:

        width: The width of the image.

        height: The height of the image.

        pixels: The total number of pixels in the image.

    Returns:

        The resolution of the image as a string in the format of "{X}x{Y}", where {X} is the width and {Y} is the height.

    """

    resolution = round(pixels / (width * height))

    return f"{resolution}x{resolution}"

