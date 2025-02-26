import re

from collections import Counter

from typing import List



def find_top_k_words(sentences: List[str], k: int) -> List[tuple]:

    """Finds the top k most frequently used words in a list of sentences.



    Args:

        sentences: A list of sentences, where each sentence is a string.

        k: The number of top words to return.



    Returns:

        A list of tuples, where each tuple contains a word and its frequency.

    """

    word_counts = Counter()

    pattern = re.compile(r'\W+')

    for sentence in sentences:

        words = pattern.split(sentence)

        words = [word for word in words if word]

        for word in words:

            word_counts[word] += 1

    return word_counts.most_common(k)

