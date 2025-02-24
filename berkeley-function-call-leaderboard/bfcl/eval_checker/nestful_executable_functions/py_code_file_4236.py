from typing import Union



def bit_string_to_int(bit_string: str) -> Union[int, None]:

    """Converts a string of 1s and 0s to its integer representation.



    Args:

        bit_string: A string of 1s and 0s.



    Returns:

        The integer representation of the input string, or None if the input string contains non-binary characters.

    """

    int_value = 0

    for i, digit in enumerate(bit_string):

        if digit not in '01':

            return None

        digit_value = ord(digit) - ord('0')

        weight = 2 ** (len(bit_string) - i - 1)

        int_value += digit_value * weight



    return int_value

