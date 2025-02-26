from typing import Tuple



def wrapped_coordinates(coordinates: Tuple[int, int], image_width: int, image_height: int) -> Tuple[int, int]:

    """

    Wraps a given coordinate to fit within the range of the image.



    Args:

        coordinates: The original coordinate (x, y).

        image_width: The width of the image.

        image_height: The height of the image.



    Returns:

        The wrapped coordinate (x', y').

    """

    x, y = coordinates

    wrapped_x = x % image_width

    wrapped_y = y % image_height



    return wrapped_x, wrapped_y

