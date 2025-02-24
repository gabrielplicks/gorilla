from typing import Union



def add_leading_zeros(string: Union[str, int]) -> str:

    """Adds two leading zeros to the input string if its length is less than 2.



    Args:

        string: The input string.



    Returns:

        The input string with two leading zeros if its length is less than 2,

        otherwise the original input string.

    """

    if len(str(string)) < 2:

        string = "00" + str(string)

    return string

