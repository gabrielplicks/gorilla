from typing import List, Union



def sum_every_second(lst: List[Union[int, str]]) -> int:

    """Calculates the sum of every second element in a list, provided that the element is an integer.

    If the list is empty or all elements are not integers, returns 0.

    Args:

        lst: The list of elements.

    """

    if not lst:

        return 0

    sum_ = 0

    for i, elem in enumerate(lst):

        if isinstance(elem, int) and i % 2 == 0:

            sum_ += elem

    return sum_

