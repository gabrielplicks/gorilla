from typing import List



def string_join(words: List[str]) -> str:

    """Joins a list of words into a single string, separated by a single space.



    Args:

        words: A list of words to join together.



    Returns:

        A single string with the words joined together, separated by a single space.

    """

    return ''.join([word + ' ' for word in words])

