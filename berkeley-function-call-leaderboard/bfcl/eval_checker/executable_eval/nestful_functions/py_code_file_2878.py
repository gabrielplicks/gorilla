from typing import Dict, List



def words_to_dict(words: List[str]) -> Dict[str, int]:

    """Converts a list of words to a dictionary where each key is a word and the value is the number of times it appears in the list.



    Args:

        words: A list of words.



    Returns:

        A dictionary where each key is a word and the value is the number of times it appears in the list.

    """

    word_counts = {}

    for word in words:

        word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts

