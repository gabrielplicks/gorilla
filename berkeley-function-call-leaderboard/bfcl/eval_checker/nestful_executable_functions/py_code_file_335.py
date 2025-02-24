from typing import Optional



def first_non_letter(input_string: str) -> Optional[str]:

    """Returns the first non-letter character in the input string.

    If the input string has only letters, the function returns None.

    Args:

        input_string: The input string to search for the first non-letter character.

    """

    for char in input_string:

        if not char.isalpha():

            return char

    return None

