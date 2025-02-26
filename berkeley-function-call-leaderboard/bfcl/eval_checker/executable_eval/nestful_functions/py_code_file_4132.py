from itertools import combinations

from typing import List, Tuple



def count_ngrams(words: List[str], n: int) -> int:

    """Counts the number of unique N-grams in a given list of words.



    Args:

        words: A list of words.

        n: The desired N-gram length.



    Returns:

        The total number of unique N-grams.

    """

    ngrams = set()

    for i in range(len(words) - n + 1):

        ngrams.add(tuple(words[i:i+n]))

    return len(ngrams)

