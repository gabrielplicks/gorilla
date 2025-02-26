from typing import List, Tuple



def sort_by_combined_length(tuples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

    """Sorts a list of tuples based on their combined lengths.



    Each tuple has two string elements, and the combined length of a tuple is the sum of the lengths of its two string elements. The sorting is done first by the length of the first element in each tuple, then by the length of the second element.



    Args:

        tuples: A list of tuples, where each tuple has two string elements.



    Returns:

        A sorted list of tuples.

    """

    def key_func(tuple: Tuple[str, str]) -> int:

        return len(tuple[0]) + len(tuple[1])



    return sorted(tuples, key=key_func)

