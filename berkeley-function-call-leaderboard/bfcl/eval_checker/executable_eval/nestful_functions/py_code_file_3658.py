from collections import Counter

from typing import Dict



def top_k_words(sentence: str, k: int) -> Dict[str, int]:

    """Returns a dictionary of the top `k` most frequently occurring words in a given string.



    Args:

        sentence: A string representing a sentence or passage to be analyzed.

        k: An integer representing the number of most frequent words to return.



    Returns:

        A dictionary with the top `k` most frequently occurring words as keys and their respective frequencies as values.

    """

    word_counts = Counter(sentence.split())

    return {word: count for word, count in word_counts.most_common(k)}

