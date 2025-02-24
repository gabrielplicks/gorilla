from typing import Union



def map_str_to_int(s: str) -> Union[int, AssertionError]:

    """Maps a string to an integer. If the string is invalid, the function raises an AssertionError.



    Args:

        s: The string to be mapped to an integer.



    Raises:

        AssertionError: If the string is invalid.

    """

    if not s.isdigit():

        raise AssertionError('Invalid string: Not all characters are digits')

    return sum(int(digit) for digit in s)

