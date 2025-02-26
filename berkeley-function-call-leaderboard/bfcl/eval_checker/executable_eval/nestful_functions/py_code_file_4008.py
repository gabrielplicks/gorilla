from typing import Set



def escape_markdown(text: str) -> str:

    """Escapes all Markdown special characters in a given text.



    Args:

        text: The input text.



    Returns:

        The escaped text.

    """

    special_characters: Set[str] = set(['*', '_', '#', '\\'])

    escaped_text: str = ''



    for character in text:

        if character in special_characters:

            escaped_text += '\\'

        escaped_text += character



    return escaped_text

