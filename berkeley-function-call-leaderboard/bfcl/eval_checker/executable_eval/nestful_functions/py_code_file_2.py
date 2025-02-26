from typing import List



def find_max_for_each_row(A: List[List[float]]) -> List[float]:

    """Finds the maximum value for each row in a 2D array.



    Args:

        A: A 2D array with dimensions m x n.



    Returns:

        A 1D array with dimensions m x 1, where each element is the maximum value of the corresponding row in the input array.

    """

    m, n = len(A), len(A[0])

    max_values = []

    for i in range(m):

        max_value = A[i][0]

        for j in range(1, n):

            if A[i][j] > max_value:

                max_value = A[i][j]

        max_values.append(max_value)

    return max_values

