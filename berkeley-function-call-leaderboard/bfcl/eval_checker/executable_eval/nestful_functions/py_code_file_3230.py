import re

from typing import List, Union



def format_title_case(text: str) -> str:

    """Formats a string to title case, removing leading and trailing spaces.



    Args:

        text: The input string to be formatted.



    Returns:

        The formatted string in title case.

    """

    text = text.strip()

    words: List[str] = re.findall(r'\b\S+', text)

    formatted_words: List[str] = [word.title() for word in words]

    return ' '.join(formatted_words)

