from typing import Dict



def get_most_frequent(word_freq: Dict[str, int], k: int) -> list:

    """Gets the most frequent `k` words from a dictionary mapping words to their frequency of occurrence.

    Args:

        word_freq: A dictionary mapping words to their frequency of occurrence.

        k: The number of most frequent words to return.

    """

    sorted_dict = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

    frequent_words = [x[0] for x in sorted_dict.items()][:k]

    return frequent_words

