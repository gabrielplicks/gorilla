from typing import List, Tuple



def combine_lists_into_tuples(l1: List[int], l2: List[int]) -> List[Tuple[int, int]]:

    """Combines two lists into a list of tuples.



    Args:

        l1: The first list.

        l2: The second list.



    Returns:

        A list of tuples, where each tuple contains one element from each list.

    """

    return [(a, b) for a, b in zip(l1, l2)]

