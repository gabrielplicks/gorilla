from typing import List, Dict



def word_length_dict(words: List[str]) -> Dict[str, int]:

    """Creates a dictionary that maps each word to its length.



    Args:

        words: A list of words.



    Returns:

        A dictionary that maps each word to its length.

    """

    return dict(zip(words, [len(word) for word in words]))

