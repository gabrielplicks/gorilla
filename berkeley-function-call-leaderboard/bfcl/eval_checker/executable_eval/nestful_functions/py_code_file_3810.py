from typing import List



def reverse_words_in_string(string: str) -> str:

    """Reverses the order of words in a string.



    Args:

        string: The input string.



    Returns:

        A new string with the order of words reversed.

    """

    words: List[str] = string.split()

    words.reverse()

    return ' '.join(words)

