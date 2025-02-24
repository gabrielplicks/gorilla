from typing import List, Tuple



def sum_tuple(t: Tuple[int, int]) -> int:

    """Calculates the sum of the first and second elements of a tuple.

    Args:

        t: The tuple.

    """

    return t[0] + t[1]



def sort_tuples(tuples: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    """Sorts a list of tuples based on the sum of the first and second elements of each tuple.

    Args:

        tuples: The list of tuples.

    """

    return sorted(tuples, key=sum_tuple)

