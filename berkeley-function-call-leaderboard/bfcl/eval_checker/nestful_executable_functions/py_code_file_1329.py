from typing import Union



def capitalize_string(string: str, only_first_letter: bool = False) -> str:

    """Capitalizes a string based on the boolean argument.



    Args:

        string: The string to capitalize.

        only_first_letter: A boolean indicating whether to capitalize only the first letter or all the letters.



    Returns:

        The capitalized string.

    """

    if only_first_letter:

        return string.capitalize()

    else:

        return string.upper()

