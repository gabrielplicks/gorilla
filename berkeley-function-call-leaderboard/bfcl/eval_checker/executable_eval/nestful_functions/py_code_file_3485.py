import re

from typing import Match



def remove_curly_braces(text: str) -> str:

    """Removes curly braces from a string and replaces them with the text between them.



    Args:

        text: The input string.



    Returns:

        The input string with curly braces removed and replaced by the text between them.

    """

    pattern = re.compile(r'\{(.*?)\}')

    matches: Match[str] = pattern.finditer(text)

    for match in matches:

        text = text.replace(match.group(), match.group(1))

    return text

