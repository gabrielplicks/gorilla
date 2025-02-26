from typing import List



def average_of_lists(lists: List[List[float]]) -> float:

    """Calculates the average of all numbers in a list of lists.



    Args:

        lists: A list of lists of numbers.



    Returns:

        The average of all numbers in the lists.

    """

    flattened_list = []

    for sublist in lists:

        flattened_list.extend(sublist)

    return sum(flattened_list) / len(flattened_list)

