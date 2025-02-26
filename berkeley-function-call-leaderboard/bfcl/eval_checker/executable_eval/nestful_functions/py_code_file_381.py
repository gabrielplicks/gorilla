from typing import Union



def convert_string_to_number(s: str) -> Union[int, float]:

    """Converts a string representing a non-negative integer to its corresponding integer value.



    Args:

        s: The string to be converted.



    Returns:

        The integer value of the string.

    """

    result = 0

    for i, c in enumerate(s[::-1]):

        result += int(c) * 10 ** i

    return result

