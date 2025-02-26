from typing import List



def transpose_if_true(array: List[List[int]], transpose: bool = False) -> List[List[int]]:

    """

    Transpose the given array if the 'transpose' parameter is 'True', otherwise return the same array.

    If the 'transpose' parameter is 'True', the returned array should be in transposed form,

    meaning the rows become columns and columns become rows.



    Args:

        array: A 2D array to be transposed.

        transpose: A boolean indicating whether to transpose the array or not.

    """

    if transpose:

        return list(map(list, zip(*array)))

    return array

