from typing import Tuple



def convert_to_rgb(color_hex: str) -> Tuple[int, int, int]:

    """Converts a color code in the #RRGGBB format to the corresponding RGB color code as a tuple of three integers in the range of 0 to 255 inclusive.



    Args:

        color_hex: The color code in the #RRGGBB format.



    Returns:

        A tuple of three integers representing the RGB color code.

    """

    color_hex = color_hex.lstrip("#")

    r = int(color_hex[:2], 16)

    g = int(color_hex[2:4], 16)

    b = int(color_hex[4:], 16)

    return (r, g, b)

