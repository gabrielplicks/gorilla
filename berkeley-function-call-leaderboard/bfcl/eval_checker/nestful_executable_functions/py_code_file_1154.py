from typing import List



def reverse_words_in_a_string(string: str) -> str:

    """

    Reverses the order of words in a string while maintaining the order of letters within each word.



    Args:

        string: The input string.



    Returns:

        The input string with the order of words reversed.

    """

    words: List[str] = string.split()

    words.reverse()

    return ' '.join(words)

