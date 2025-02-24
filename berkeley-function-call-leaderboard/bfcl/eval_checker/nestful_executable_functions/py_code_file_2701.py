from typing import Union



def concatenate_one_by_one(string: Union[str, bytes]) -> str:

    """Concatenates a string one character at a time using a for loop.



    Args:

        string: The input string to be concatenated.



    Returns:

        The concatenated string.

    """

    concatenated_string = ""

    for character in string:

        concatenated_string += character

    return concatenated_string

