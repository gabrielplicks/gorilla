from typing import List, Union



def is_stochastic_matrix(matrix: List[List[Union[int, float]]]) -> Union[float, None]:

    """Checks if a two-dimensional list of numbers is a valid stochastic matrix.

    A stochastic matrix is a square matrix where each row sums to 1.0.

    Args:

        matrix: A two-dimensional list of numbers.

    Returns:

        If the input matrix is valid, returns the sum of all elements in the matrix.

        If the input matrix is invalid, returns None.

    """

    if len(matrix) != len(matrix[0]):

        return None

    for row in matrix:

        if sum(row) != 1.0:

            return None

    sum_elements = sum(sum(row) for row in matrix)



    return sum_elements

