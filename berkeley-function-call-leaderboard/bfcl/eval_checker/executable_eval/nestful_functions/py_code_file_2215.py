import re

from typing import List, Union



def say_bad(text: str) -> str:

    """

    Replaces the word "bad" in a given string with an empty string.

    If the word "bad" is contained within a larger word, only that word is modified.



    Args:

        text: The input string.



    Returns:

        The modified string.

    """

    words: List[str] = text.split(' ')

    for i, word in enumerate(words):

        if re.search(r'\bbad\b', word):

            words[i] = re.sub(r'\bbad\b', '', word)

    return ' '.join(words)

