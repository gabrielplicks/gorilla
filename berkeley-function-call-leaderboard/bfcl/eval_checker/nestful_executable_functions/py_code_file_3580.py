from typing import List, Tuple



def sum_tuples_in_lists(list1: List[Tuple[int, int, int]], list2: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:

    """Sum the tuples in each list.



    Args:

        list1: A list of tuples.

        list2: A list of tuples.



    Returns:

        A list of tuples that represent the sum of each tuple in `list1` and `list2`.

    """

    result = []

    for i in range(len(list1)):

        sum_tuple = tuple(a + b for a, b in zip(list1[i], list2[i]))

        result.append(sum_tuple)

    return result

