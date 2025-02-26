from typing import List



def transpose_2d_array(array: List[List[float]]) -> List[List[float]]:

    """Transposes a 2D array using list comprehension.



    Args:

        array: The input 2D array.



    Returns:

        The transposed 2D array.

    """

    rows = len(array)

    cols = len(array[0])



    return [[array[j][i] for j in range(rows)] for i in range(cols)]

