from typing import List



def hex_code_to_rgb(hex_code: str) -> List[int]:

    """Converts a hexadecimal color code to a list of RGB values.



    Args:

        hex_code: A string representing a hexadecimal color code.



    Returns:

        A list of integers representing the RGB values.

    """

    hex_code = hex_code.lstrip('#')

    rgb = [int(hex_code[i:i+2], 16) for i in range(0, len(hex_code), 2)]



    return rgb

