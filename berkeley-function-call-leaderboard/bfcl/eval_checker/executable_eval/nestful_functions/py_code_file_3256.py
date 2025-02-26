import re

import typing



def extract_special_characters(string: str) -> typing.List[str]:

    """Extracts all special characters from a string and returns them as a list.

    Special characters in this case are those that are not alphanumeric, including any punctuation marks, symbols, or whitespaces.

    Args:

        string: The input string.

    Returns:

        A list of special characters.

    """

    try:

        return re.findall(r'[^a-zA-Z0-9\s]', string)

    except Exception:

        return []

