from typing import Union



def parse_boolean(text: str) -> Union[bool, None]:

    """Parses a boolean value from a string.



    Args:

        text: The string to parse.



    Returns:

        True if the string is "true" (case-insensitive), False if the string is "false", and None otherwise.

    """

    text = text.lower()

    if text == "true":

        return True

    elif text == "false":

        return False

    else:

        return None

