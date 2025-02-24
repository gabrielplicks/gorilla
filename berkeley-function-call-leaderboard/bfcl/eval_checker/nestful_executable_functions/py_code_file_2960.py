from typing import Union



def titleize(text: Union[str, bytes]) -> str:

    """Converts the first character of each word in a string to uppercase and the rest to lowercase.

    For example, titleize("a simple string") should return "A Simple String". titleize("A simple string")

    should return "A Simple String". If the input string is not in Unicode, it should be encoded as UTF-8 before

    processing.

    Args:

        text: The input string to be formatted.

    """

    if not isinstance(text, str):

        text = text.decode("utf-8")

    return text.title()

