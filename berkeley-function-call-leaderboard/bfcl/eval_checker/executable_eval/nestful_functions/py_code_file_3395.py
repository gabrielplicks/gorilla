from typing import Union



def convert_positive_int_to_binary_string(number: Union[int, float]) -> str:

    """Converts a positive integer to a binary string representation.



    Args:

        number: The input number.



    Raises:

        ValueError: If the input number is negative.

    """

    if number < 0:

        raise ValueError("The input number must be positive.")

    return bin(int(number))[2:]  # Strip the "0b" prefix

