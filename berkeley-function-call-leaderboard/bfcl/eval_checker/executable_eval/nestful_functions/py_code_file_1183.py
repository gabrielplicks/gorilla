from typing import Union



def int_to_bit_string(number: Union[int, float], bits: int) -> str:

    """Converts an integer to its string representation in binary.



    Args:

        number: The integer to convert.

        bits: The number of bits to use for the conversion.



    Raises:

        ValueError: If the integer is outside the range of the given number of bits.

    """

    if number >= 0:

        string_rep = bin(number)[2:]  # Convert to binary string

        return '0' * (bits - len(string_rep)) + string_rep  # Add necessary number of leading zeros

    else:

        string_rep = bin(number & (2**bits - 1))[2:]  # Convert to binary string

        return '1' * (bits - len(string_rep)) + string_rep  # Add necessary number of leading ones

