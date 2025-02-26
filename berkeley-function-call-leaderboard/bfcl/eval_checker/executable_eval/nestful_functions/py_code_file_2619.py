from typing import List



def reverse_words_order(string: str) -> str:

    """Reverses the order of words in a string.



    Args:

        string: The input string.



    Returns:

        A string with the same words in the reverse order.

    """

    words: List[str] = string.split()

    return ' '.join(reversed(words))

