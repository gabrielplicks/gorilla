from typing import List



def increment_array_in_place(array: List[List[int]]) -> List[List[int]]:

    """Increments each element in the given array by 1 in-place.



    Args:

        array: The input 2D array.



    Returns:

        The modified array.

    """

    for row in array:

        for i in range(len(row)):

            row[i] += 1

    return array

