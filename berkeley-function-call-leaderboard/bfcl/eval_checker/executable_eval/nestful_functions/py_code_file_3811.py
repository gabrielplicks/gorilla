from typing import Union



def format_word(word: Union[str, None]) -> str:

    """Formats a word to have a specific style.



    Args:

        word: The word to format.



    Returns:

        The formatted word.

    """

    if len(word) == 1:

        return word.upper()

    elif len(word) > 1:

        if word.isupper():

            return word.lower()

        else:

            return word.capitalize()

    else:

        return ''

