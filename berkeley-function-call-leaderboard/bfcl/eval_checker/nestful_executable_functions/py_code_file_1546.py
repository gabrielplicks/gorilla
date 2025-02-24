from typing import List, Tuple



def sort_tuples_by_second_value(tuples: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    """Sorts a list of tuples by the second value in descending order.



    The returned sorted list contains tuples of number pairs, and the function uses the first value of each pair as the sorting key. If two tuples have the same first values, the function uses the second value as the tie-breaker.



    Args:

        tuples: A list of tuples to be sorted.



    Returns:

        A sorted list of tuples.

    """

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

