from typing import Tuple



def convert_rgba_to_hex(rgba: Tuple[int, int, int, int]) -> str:

    """Converts a color represented in RGBA tuple (red, green, blue, alpha) to a hexadecimal string in the format `#RRGGBB` (without the alpha channel).



    Args:

        rgba: The RGBA tuple representing the color.



    Returns:

        A string with length 7 representing the hexadecimal color.

    """

    hex_red = format(rgba[0], '02x')

    hex_green = format(rgba[1], '02x')

    hex_blue = format(rgba[2], '02x')

    hex_color = '#' + hex_red + hex_green + hex_blue

    return hex_color

