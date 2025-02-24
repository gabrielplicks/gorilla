from typing import Union



def convert_bin_to_dec(binary_string: Union[str, bytes]) -> int:

    """Converts an 8-bit binary string to a decimal integer.



    Args:

        binary_string: The binary string to convert.



    Raises:

        ValueError: If the binary string contains any characters other than '0' and '1'.

    """

    if any(char not in '01' for char in binary_string):

        raise ValueError("Not a binary string")



    decimal_int = 0

    for i, char in enumerate(reversed(binary_string)):

        decimal_int += int(char) * (2 ** i)



    return decimal_int

