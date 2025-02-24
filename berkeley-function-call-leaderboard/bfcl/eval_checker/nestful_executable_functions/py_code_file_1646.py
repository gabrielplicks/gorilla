from typing import List, Tuple



def get_value_counts(values: List[int]) -> List[Tuple[int, int]]:

    """

    Builds a list of tuples, where each tuple represents a distinct value in the input list with the number of its occurrences.



    Args:

        values: A list of integers.



    Returns:

        A list of tuples, where each tuple represents a distinct value in the input list with the number of its occurrences.

    """

    value_counts = {}



    for value in values:

        if value in value_counts:

            value_counts[value] += 1

        else:

            value_counts[value] = 1



    return list(value_counts.items())

