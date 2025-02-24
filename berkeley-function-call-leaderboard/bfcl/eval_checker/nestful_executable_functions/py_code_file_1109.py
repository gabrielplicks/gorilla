from typing import AnyStr



def get_meaningful_chars(string: AnyStr) -> AnyStr:

    """

    Returns a new string that contains all the meaningful (alphanumeric and whitespace) characters from the original string.



    Args:

        string: The input string.

    """

    new_string = ''

    for char in string:

        if char.isalnum() or char.isspace():

            new_string += char

    return new_string

