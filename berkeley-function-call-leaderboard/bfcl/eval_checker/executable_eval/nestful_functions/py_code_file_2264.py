from typing import List



def add_previous(lst: List[int]) -> List[int]:

    """Creates a new list where each element is the sum of the current element and the previous element in the original list.

    Args:

        lst: A list of integers.

    """

    sum_previous = lambda lst, i: lst[i] + lst[i-1]

    if len(lst) == 0:

        return []

    return list(map(lambda i: sum_previous(lst, i), range(1, len(lst))))

