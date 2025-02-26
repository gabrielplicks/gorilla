from typing import Union



def decimal_to_bitstring(decimal: Union[int, float]) -> str:

    """Converts a decimal number to its 8-bit binary representation.



    Args:

        decimal: The decimal number to be converted.



    Returns:

        The 8-bit binary representation of the decimal number as a string.

    """

    mask = 0b11111111

    binary = decimal & mask

    binary_string = bin(binary)[2:]

    return binary_string.zfill(8)

