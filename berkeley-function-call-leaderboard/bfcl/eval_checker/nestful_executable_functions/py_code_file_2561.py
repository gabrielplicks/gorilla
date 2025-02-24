from typing import Dict



def most_common_char(text: str) -> str:

    """Finds the most common character in a string.



    Args:

        text: The input string.



    Returns:

        The most common character in the string. If there are multiple characters

        that appear the same number of times, the function returns the character

        that appears first in the string.

    """

    char_counts: Dict[str, int] = {char: text.count(char) for char in text}

    most_common_char = max(char_counts, key=char_counts.get)

    return most_common_char

