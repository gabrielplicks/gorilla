from collections import Counter

from typing import Dict, List



def count_words_in_words(words: List[str]) -> Dict[str, int]:

    """Counts the number of occurrences of each word in a list of words.



    Args:

        words: A list of words.



    Returns:

        A dictionary with words as keys and word counts as values. The dictionary

        only includes words that appear more than once in the list.

    """

    counter = Counter(words)

    return {word: count for word, count in counter.items() if count > 1}

