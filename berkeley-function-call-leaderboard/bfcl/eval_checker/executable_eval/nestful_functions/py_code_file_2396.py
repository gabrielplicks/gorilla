from typing import Tuple



def get_hex_to_rgb(hex_color: str) -> str:

    """Converts a hexadecimal color code to its corresponding RGB values.

    Args:

        hex_color: A hexadecimal color code.

    Returns:

        A string formatted as "(R, G, B)" with the RGB values separated by commas.

    """

    hex_color = hex_color.lstrip('#')

    red = int(hex_color[:2], base=16)

    green = int(hex_color[2:4], base=16)

    blue = int(hex_color[4:], base=16)

    return '({}, {}, {})'.format(red, green, blue)

