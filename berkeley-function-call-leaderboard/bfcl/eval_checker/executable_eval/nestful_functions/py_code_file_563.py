from typing import Union



def first_number(s: str) -> Union[int, None]:

    """Returns the first number it encounters in the string.



    Args:

        s: The input string.



    Returns:

        The first number encountered in the string, or None if no number is found.

    """

    for c in s:

        if c.isdigit():

            return int(c)

    return None

