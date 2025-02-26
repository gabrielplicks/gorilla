import string



def convert_string_to_charmap(string: str) -> str:

    """

    Converts a string into a list of characters using the `charmap` encoding.



    Args:

        string: The input string to be converted.



    Returns:

        The input string converted into a list of characters using the `charmap` encoding.

    """

    charmap = string.maketrans(string, string)

    return string.translate(charmap)

