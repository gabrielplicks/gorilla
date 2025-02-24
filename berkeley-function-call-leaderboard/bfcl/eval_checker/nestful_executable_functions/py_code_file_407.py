from typing import Union



def convert_decimal_to_binary(num: Union[int, float]) -> str:

    """Converts a decimal number to its binary representation as a string using bitwise operations.

    Args:

        num: The decimal number to convert.

    """

    binary_repr = ''



    while num > 0:

        bit = num & 1

        binary_repr = str(bit) + binary_repr

        num = num >> 1



    return binary_repr

