from typing import Union



def num_value(string: str) -> int:

    """Converts a string into an integer.



    The function handles negative integers, float numbers, and numbers represented in scientific notation.

    If the input cannot be converted to an integer, the function returns 0.



    Args:

        string: The input string to be converted into an integer.



    Returns:

        An integer representing the numerical value of the input string.

    """

    try:

        return int(float(string))

    except:

        return 0

