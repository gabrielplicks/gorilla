from typing import List



def reverse_english_words(text: str) -> str:

    """Reverses the order of words in an English string.



    Args:

        text: The input string.



    Returns:

        The input string with the order of words reversed.

    """

    return ' '.join(reversed(text.split()))

