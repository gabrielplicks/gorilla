from typing import List



def group_by_modular_arithmetic(list_of_integers: List[int], number_of_groups: int) -> List[List[int]]:

    """Groups a list of integers into a specified number of groups based on modular arithmetic.

    Args:

        list_of_integers: A list of integers to be grouped.

        number_of_groups: The number of groups to divide the integers into.

    """

    if not list_of_integers:

        return []

    result = [[] for _ in range(number_of_groups)]

    for i, num in enumerate(list_of_integers):

        group_index = i % number_of_groups

        result[group_index].append(num)

    return result

