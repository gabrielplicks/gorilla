from typing import Union



def convert_binary_to_decimal(binary_string: str) -> int:

    """Converts a binary string into a decimal integer.



    Args:

        binary_string: A binary string to be converted into a decimal integer.



    Returns:

        The decimal integer representation of the binary string.

    """

    decimal = 0

    for bit in binary_string:

        decimal <<= 1

        if bit == '1':

            decimal += 1

    return decimal



def convert_binary_to_decimal_alternative(binary_string: Union[str, int]) -> int:

    """Converts a binary string or integer into a decimal integer.



    Args:

        binary_string: A binary string or integer to be converted into a decimal integer.



    Returns:

        The decimal integer representation of the binary string or integer.

    """

    return int(binary_string, base=2)

