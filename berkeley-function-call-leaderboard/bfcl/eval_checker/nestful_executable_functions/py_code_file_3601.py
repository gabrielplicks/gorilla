import re

from typing import Dict, Any



def modify_text(text: str) -> str:

    """Modifies a string by replacing all occurrences of the words "foo", "bar", and "baz" with "bar", "baz", and "qux", respectively.



    Args:

        text: The input string to be modified.



    Returns:

        The modified string.

    """

    word_map: Dict[str, Any] = {'foo': 'bar', 'bar': 'baz', 'baz': 'qux'}

    pattern = re.compile(r'\b(?:foo|bar|baz)\b')

    return pattern.sub(lambda match: word_map[match.group()], text)

