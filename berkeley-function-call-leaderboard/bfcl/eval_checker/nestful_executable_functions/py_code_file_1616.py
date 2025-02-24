from typing import Dict, List, Tuple



def word_count_to_list(word_count_dict: Dict[str, int]) -> List[Tuple[str, int]]:

    """Transforms a dictionary of word counts into a list of pairs.



    Each pair represents a unique word and its frequency.



    Args:

        word_count_dict: A dictionary where keys are words and values are their frequencies.



    Returns:

        A list of tuples. Each tuple contains a word and its frequency.

    """

    word_count_list = []

    for word, count in word_count_dict.items():

        word_count_list.append((word, count))

    return word_count_list

