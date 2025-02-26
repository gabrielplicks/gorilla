from typing import List, Union



def compute_mean(two_d_list: List[List[Union[int, float]]]) -> float:

    """Computes the mean of a two-dimensional list of integers or floats.



    Args:

        two_d_list: A two-dimensional list of integers or floats.



    Returns:

        The mean of the elements in the two-dimensional list.

    """

    flattened_list = [item for sublist in two_d_list for item in sublist]

    total = sum(flattened_list)

    n = len(flattened_list)

    return total / n

