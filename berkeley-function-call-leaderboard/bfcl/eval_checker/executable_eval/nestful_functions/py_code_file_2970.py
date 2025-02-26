from typing import Dict, List



def index_words(words: List[str]) -> Dict[str, int]:

    """Generates a dictionary mapping from each word in a list to its index in the list.



    Args:

        words: A list of words.



    Returns:

        A dictionary where each word is mapped to its index in the list.

    """

    index_map = {}

    for index, word in enumerate(words):

        index_map[word] = index



    return index_map

