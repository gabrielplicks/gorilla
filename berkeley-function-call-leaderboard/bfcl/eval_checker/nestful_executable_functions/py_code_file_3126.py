from typing import Union



def change_language_code(input_string: str) -> str:

    """Changes the language code from "en" to "de" or "fr" based on the input.

    If the input string starts with "en", change it to "de" and return the new string.

    If the input string starts with "de" or "fr", change it to "en" and return the new string.

    If the input string does not start with "en", "de", or "fr", raise a ValueError.

    Args:

        input_string: The input string to change the language code.

    """

    if input_string.startswith('en'):

        return input_string.replace('en', 'de')

    elif input_string.startswith('de') or input_string.startswith('fr'):

        return input_string.replace('de', 'en').replace('fr', 'en')

    else:

        raise ValueError('Invalid input string')

