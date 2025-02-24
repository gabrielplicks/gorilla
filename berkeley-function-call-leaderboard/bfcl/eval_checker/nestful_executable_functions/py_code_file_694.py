from typing import Tuple



def clip_point(point: Tuple[int, int], width: int, height: int) -> Tuple[int, int]:

    """Clips a point to the boundaries of an image.



    Args:

        point: The point to clip.

        width: The width of the image.

        height: The height of the image.



    Raises:

        Exception: If the point is out of bounds.

    """

    x, y = point

    if x < 0 or x >= width or y < 0 or y >= height:

        raise Exception("Point out of bounds")

    return (x, y)

