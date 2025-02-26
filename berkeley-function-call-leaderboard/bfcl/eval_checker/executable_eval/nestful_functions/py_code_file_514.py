from typing import Union



def generic_to_binary_string(value: Union[int, str]) -> str:

    """Converts a generic value (integer or string) to a binary string representation, with leading zeros to ensure a length of 8 bits.



    Args:

        value: The value to convert to binary string representation.



    Raises:

        ValueError: If the value is not a valid integer or string.

    """

    if isinstance(value, int):

        binary_string = bin(value)[2:]

    elif isinstance(value, str):

        binary_string = bin(ord(value))[2:]

    else:

        raise ValueError("Value must be an integer or string.")

    zero_padding = "0" * (8 - len(binary_string))

    binary_string = zero_padding + binary_string



    return binary_string

