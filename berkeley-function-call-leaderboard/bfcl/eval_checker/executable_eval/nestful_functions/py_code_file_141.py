from typing import List



def int_to_rgba(value: int) -> List[int]:

    """Converts an integer value to the corresponding RGBA (red, green, blue, alpha) color components.



    Args:

        value: A positive integer representing the color value.



    Returns:

        A list of four integers representing the RGBA color components.

    """

    value = value & 0xFFFFFFFF

    b = value & 0xFF

    g = (value >> 8) & 0xFF

    r = (value >> 16) & 0xFF

    a = (value >> 24) & 0xFF

    rgba = [r, g, b, a]



    return rgba

