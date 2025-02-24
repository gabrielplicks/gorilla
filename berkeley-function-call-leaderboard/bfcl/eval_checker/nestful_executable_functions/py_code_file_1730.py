from typing import Union



def add_strings_with_graceful_error_handling(string1: str, string2: str) -> Union[int, str]:

    """Adds two string integers and returns the result. If the input strings are not valid integers, the function returns the string "ERROR".



    Args:

        string1: The first string integer.

        string2: The second string integer.



    Returns:

        The sum of the two string integers, or the string "ERROR" if the input strings are not valid integers.

    """

    try:

        num1 = int(string1)

        num2 = int(string2)

    except ValueError:

        return "ERROR"

    return num1 + num2

