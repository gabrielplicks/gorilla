from typing import Dict



def sort_freq_dist(freq_dist: Dict[str, int]) -> list:

    """Sorts the entries of a frequency distribution of words in descending order using a Python dictionary.



    Args:

        freq_dist: A dictionary where the keys are words and the values are positive integers representing the frequencies of those words.



    Returns:

        A list of the words sorted in descending order of their frequencies.

    """

    return sorted(freq_dist, key=freq_dist.get, reverse=True)

