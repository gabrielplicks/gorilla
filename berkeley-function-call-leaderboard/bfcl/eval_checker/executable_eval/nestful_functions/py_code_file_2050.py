from typing import List



def remove_words_with_digit(words: List[str]) -> List[str]:

    """Removes words with digits from a list of words.



    Args:

        words: A list of words.



    Returns:

        A new list with all words that contain a digit removed.

    """

    return [word for i, word in enumerate(words) if not any(digit in word for digit in '0123456789')]

