from typing import Tuple



def pixel_position_to_angle(x: int, width: int) -> float:

    """Converts a pixel position `x` to an angle in degrees.



    The angle is in the range `-90` to `90`, where `-90` maps to the leftmost pixel and `90` maps to the rightmost pixel.



    Args:

        x: The pixel position.

        width: The image width in pixels.



    Returns:

        The angle in degrees.

    """

    pixels_per_side = width // 2

    ratio = 90 / pixels_per_side

    pixels_relative_to_center = x - pixels_per_side

    angle = ratio * pixels_relative_to_center



    return angle

