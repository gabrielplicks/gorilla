from typing import List, Tuple



def minimum_maximum_of_sublists(lst: List[int]) -> List[Tuple[int, int]]:

    """Calculates the minimum and maximum values of each sublist of 3 consecutive elements in a given list.



    Args:

        lst: A list of integers.



    Returns:

        A list of tuples containing the minimum and maximum values of each sublist of 3 consecutive elements in the input list.

    """

    result = []

    for i in range(len(lst) - 2):

        sublist = lst[i:i+3]

        result.append((min(sublist), max(sublist)))

    return result

