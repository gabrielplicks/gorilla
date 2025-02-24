from typing import AnyStr



def format_javadoc_comment(text: AnyStr) -> AnyStr:

    """Formats a string with javadoc style comment formatting.



    Args:

        text: The input string to format.



    Returns:

        The formatted string.

    """

    text = text.strip().lower()

    formatted_text = '* ' + text.replace('\n', '\n* ')

    return formatted_text

