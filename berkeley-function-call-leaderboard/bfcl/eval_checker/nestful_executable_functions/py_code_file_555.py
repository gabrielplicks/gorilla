from typing import List



def transpose_array(arr: List[List[int]]) -> List[List[int]]:

    """Transposes a two-dimensional array by exchanging the rows and columns.



    Args:

        arr: A list of lists where each inner list has the same number of elements.



    Returns:

        A new list of lists where the rows and columns have been swapped.

    """

    n_rows = len(arr)

    n_cols = len(arr[0])

    new_arr = [[0 for _ in range(n_rows)] for _ in range(n_cols)]

    for i in range(n_rows):

        for j in range(n_cols):

            new_arr[j][i] = arr[i][j]

    return new_arr

