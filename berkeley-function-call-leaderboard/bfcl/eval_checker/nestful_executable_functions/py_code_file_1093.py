from typing import List



def find_sum_of_all_combinations(array1: List[int], array2: List[int]) -> List[int]:

    """Finds the sum of all possible combinations of two given integer arrays.



    Args:

        array1: A list of integers.

        array2: A list of integers.



    Returns:

        A list of integers representing the sum of each combination.

    """

    sums = []

    for i in array1:

        for j in array2:

            sums.append(i + j)

    return sums

