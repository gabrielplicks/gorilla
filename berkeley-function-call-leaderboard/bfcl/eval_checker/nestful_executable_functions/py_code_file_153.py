from typing import Union



def convert_to_sentence(string: str) -> Union[str, None]:

    """Converts a string of words into a sentence that starts with a capital letter and ends with a period.

    Args:

        string: The input string.

    Returns:

        The input string converted into a sentence format, or None if the first character is not a letter.

    """

    if string[0].isalpha():

        return string.capitalize() + '.'

    else:

        return string

