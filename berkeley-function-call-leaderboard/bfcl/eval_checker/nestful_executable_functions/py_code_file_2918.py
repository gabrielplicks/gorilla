from typing import Union



def str_title_case(string: Union[str, None]) -> str:

    """Transforms the case of a given string.

    Capitalizes the first letter of each word, and lower-case all other letters.

    A word is considered a sequence of non-space characters.

    Handles empty and whitespace-only strings by returning an empty string.

    Raises an exception for invalid input.

    Args:

        string: The input string.

    """

    if not isinstance(string, str):

        raise TypeError("Input must be a string")



    if string.isspace():

        return ""



    return string.title().replace(" ", "")

