from typing import List, Union



def normalize_casing(string: str) -> str:

    """Normalizes the casing of a string.



    Args:

        string: The input string to be normalized.



    Returns:

        The normalized string.

    """

    words: List[str] = string.split()

    normalized_words: List[str] = [word[0].upper() + word[1:].lower() for word in words]

    return " ".join(normalized_words)

