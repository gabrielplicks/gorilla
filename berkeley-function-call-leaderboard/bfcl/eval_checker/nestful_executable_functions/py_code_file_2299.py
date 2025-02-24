from typing import Union



def count_characters_unique(text: str, include_whitespace: Union[bool, None] = False) -> int:

    """Calculates the number of unique characters in a string.



    Args:

        text: The input string.

        include_whitespace: If set to True, whitespace will be included in the count.

    """

    text = text.lower().replace(" ", "")

    unique_characters = set(text)

    count = len(unique_characters)

    if include_whitespace:

        count += 1

    return count

