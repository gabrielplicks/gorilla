import string

from typing import Set



def has_common_words(a: str, b: str) -> bool:

    """Checks if two strings share a common word, ignoring punctuation, case, and Unicode characters.



    Args:

        a: The first string.

        b: The second string.



    Returns:

        True if the strings share a common word, False otherwise.

    """

    words_a: Set[str] = {word.lower() for word in a.translate(str.maketrans("", "", string.punctuation)).split()}

    words_b: Set[str] = {word.lower() for word in b.translate(str.maketrans("", "", string.punctuation)).split()}

    return bool(words_a & words_b)

