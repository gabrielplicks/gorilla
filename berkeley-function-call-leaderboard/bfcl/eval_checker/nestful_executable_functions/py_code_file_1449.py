from typing import List



def remove_short_words(text: str) -> str:

    """Removes all words with length 3 or less from a given string and returns the resulting string after splitting it by a single space.



    Args:

        text: The input string.



    Returns:

        The resulting string after removing short words.

    """

    words: List[str] = text.split(' ')

    return ' '.join([word for word in words if len(word) > 3])

