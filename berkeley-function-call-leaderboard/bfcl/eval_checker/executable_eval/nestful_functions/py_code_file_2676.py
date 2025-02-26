from collections import Counter

from typing import List



def ngram_frequency(words: List[str], n: int) -> Counter:

    """Calculates the frequency of each n-gram in a given list of words.



    An n-gram is a continuous sequence of n words.



    Args:

        words: A list of words.

        n: The size of each n-gram.



    Returns:

        A dictionary where the keys are the n-grams and the values are the corresponding frequencies.

    """

    ngrams = []

    for i in range(len(words) - n + 1):

        ngrams.append(" ".join(words[i:i+n]))

    freq_counter = Counter(ngrams)



    return freq_counter

