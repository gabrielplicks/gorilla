from typing import List



def reduce_list_recursive(l: List[float]) -> float:

    """Reduces a list of numbers to a single value by recursively applying the min function.



    Args:

        l: A list of numbers.



    Returns:

        The smallest number in the list.

    """

    if len(l) == 1:

        return l[0]

    middle_index = len(l) // 2

    first_half = l[:middle_index]

    second_half = l[middle_index:]

    reduced_first_half = reduce_list_recursive(first_half)

    reduced_second_half = reduce_list_recursive(second_half)

    return min(reduced_first_half, reduced_second_half)

