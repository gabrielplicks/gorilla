from typing import Union



def decode_binary_to_hex(binary_string: str) -> str:

    """Converts a binary string to a hexadecimal string.



    Args:

        binary_string: A string of binary digits.



    Returns:

        A string of hexadecimal digits.

    """

    hex_string = ""

    for i in range(0, len(binary_string), 4):

        group = binary_string[i:i+4]

        decimal = int(group, 2)

        hex_digit = hex(decimal)[2:]  # Strip the "0x" prefix

        hex_string += hex_digit

    return hex_string

