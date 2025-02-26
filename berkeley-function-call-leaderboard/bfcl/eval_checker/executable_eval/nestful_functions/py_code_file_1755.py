from typing import List



def words_starting_with_a(words_string: str) -> List[str]:

    """Returns a list of words that start with the letter 'a' (lower case).



    Args:

        words_string: A string of comma-separated words.

    """

    words = words_string.split(',')

    words_with_a = []

    for word in words:

        if word.startswith('a'):

            words_with_a.append(word)

    return words_with_a

