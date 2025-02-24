from typing import Union



def string_format_function(n: Union[int, float]) -> str:

    """

    Returns a string with the following format: "@" + "x" * (n - 2) + "@" if n > 1,

    otherwise returns "@".



    Args:

        n: An integer or float representing the number of characters in the string.



    Returns:

        A string with the desired format.

    """

    if n <= 1:

        return "@"

    return "@" + "x" * (n - 2) + "@"

