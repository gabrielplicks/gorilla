from typing import Union



def get_color(r: int, g: int, b: int) -> Union[str, None]:

    """

    Converts the given red, green, and blue color values to a hexadecimal color code.

    Args:

        r: The red color value.

        g: The green color value.

        b: The blue color value.

    Returns:

        A string of hexadecimal color code in the format of `#RRGGBB` if all color values are in the range of `[0, 255]`.

        Otherwise, returns `None`.

    """

    return '#' + ''.join(format(c, '02x') for c in [r, g, b]) if all(0 <= c <= 255 for c in [r, g, b]) else None

