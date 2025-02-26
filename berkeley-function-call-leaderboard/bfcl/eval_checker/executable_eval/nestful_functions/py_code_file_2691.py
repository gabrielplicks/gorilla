from typing import Union



def my_int(string_num: str) -> Union[int, None]:

    """Converts a string containing a positive integer to an integer without using the built-in int() function.

    Args:

        string_num: A string containing a positive integer.

    """

    result = 0

    for i, c in enumerate(string_num):

        result += (ord(c) - ord('0')) * (10 ** (len(string_num) - i - 1))

    return result

