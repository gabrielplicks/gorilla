from typing import Dict, List, Tuple



def get_word_indices(words: List[str]) -> Dict[str, List[int]]:

    """Creates a dictionary that maps each unique word to a list of all indices where that word occurs in the list.



    Args:

        words: A list of strings.



    Returns:

        A dictionary that maps each unique word to a list of all indices where that word occurs in the list.

    """

    return {

        word: [i for i, w in enumerate(words) if w == word]

        for word in set(words)

    }

