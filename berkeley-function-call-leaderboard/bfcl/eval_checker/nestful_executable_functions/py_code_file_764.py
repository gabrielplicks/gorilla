from typing import List



def convert_color_code(code: str) -> List[float]:

    """Converts a given color code (a string representing a hexadecimal color code) to a series of rgb values in the range [0, 1].

    The given code may or may not contain the `#` prefix, and may or may not be in all upper case letters.

    Args:

        code: The color code to be converted.

    Returns:

        A list of rgb values in the range [0, 1].

    """

    code = code.lstrip("#").upper()

    hex_code = int(code, 16)

    rgb_values = [int(code[i:i+2], 16) / 255 for i in (0, 2, 4)]

    return rgb_values

