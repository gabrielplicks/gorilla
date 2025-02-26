from typing import List



def format_message(message: str, placeholders: List[str]):

    """Formats a message by inserting a list of strings into the placeholders.



    Args:

        message: The message to be formatted.

        placeholders: A list of strings to be inserted into the placeholders.

    """

    result = message



    for placeholder, string in enumerate(placeholders):

        result = result.replace("{}", string, 1)



    return result

