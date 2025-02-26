from typing import Union



def convert_to_twos_complement(signed_int: Union[int, float]) -> int:

    """Converts a given decimal integer to its two's complement representation.



    Args:

        signed_int: The signed integer to be converted.



    Returns:

        The two's complement representation of the input as a signed integer.

    """

    binary_representation = bin(signed_int)

    negative = signed_int < 0

    max_twos_complement_value = 2**(len(binary_representation[2:]) - 1) - 1

    result = ~signed_int + 1

    if result > max_twos_complement_value:

        result -= 2 * max_twos_complement_value

    return result

