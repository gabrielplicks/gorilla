from typing import Dict



def score_words(s: str) -> Dict[str, int]:

    """Calculates the frequency of each word in a given string.



    Args:

        s: The input string containing multiple words separated by spaces.



    Returns:

        A dictionary where each key is a word from `s` and its corresponding value is the number of occurrences of that word in `s`.

    """

    words = s.split()

    word_count = {}

    for word in words:

        if word in word_count:

            word_count[word] += 1

        else:

            word_count[word] = 1

    return word_count

