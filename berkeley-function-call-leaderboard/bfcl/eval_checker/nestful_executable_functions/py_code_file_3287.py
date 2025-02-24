from typing import AnyStr



def depad(text: AnyStr) -> AnyStr:

    """Removes any leading or trailing whitespace from a string.

    Args:

        text: The input string.

    """

    return text.strip()

