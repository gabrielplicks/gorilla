from typing import Union



def extract_last_4(string: str) -> str:

    """Extracts the last 4 characters from a given string, either from the end of the string or the beginning if the string length is less than 4.



    Args:

        string: The input string.



    Returns:

        A string of exactly 4 characters.

    """

    length = len(string)

    if length < 4:

        return string[:4]

    else:

        return string[-4:]

