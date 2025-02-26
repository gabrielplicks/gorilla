from typing import AnyStr



def strip_non_alphanumeric(text: AnyStr) -> AnyStr:

    """Strips all non-alphanumeric characters from a string and replaces consecutive whitespace characters with a single space.



    Args:

        text: The input string.



    Returns:

        The processed string.

    """

    stripped_text = ''.join(ch if ch.isalnum() else ' ' for ch in text)

    stripped_text = ' '.join(stripped_text.split())



    return stripped_text

