from typing import List



def first_and_last(numbers: List[float]) -> List[float]:

    """Returns a new list containing the first and last element of the input list.

    If the input list is empty or has fewer than two elements, `first_and_last()` returns an empty list.

    Args:

        numbers: A list of numbers.

    """

    if len(numbers) < 2:

        return []



    first_and_last = [numbers[0], numbers[-1]]

    return first_and_last

