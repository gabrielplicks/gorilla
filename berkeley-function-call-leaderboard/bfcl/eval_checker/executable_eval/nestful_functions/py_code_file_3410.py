from typing import Tuple



def get_lightness(rgb: Tuple[int, int, int]) -> float:

    """Calculates the lightness value of a color in RGB space.



    Args:

        rgb: A 3-tuple of integers representing the red, green, and blue channels within the range 0 to 255.



    Returns:

        The lightness value as a floating-point number between 0 and 1.

    """

    r, g, b = rgb

    max_value = max(r, g, b)

    min_value = min(r, g, b)

    lightness = (max_value + min_value) / 2

    return lightness

