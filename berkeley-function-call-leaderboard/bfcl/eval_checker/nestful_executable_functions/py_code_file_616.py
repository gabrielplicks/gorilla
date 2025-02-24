from typing import List, Tuple



def size(matrix: List[List[float]]) -> Tuple[int, int]:

    """Finds the size of a matrix as a tuple `(height, width)`.



    Args:

        matrix: The input matrix.



    Returns:

        The size of the matrix as a tuple `(height, width)`.

    """

    height = len(matrix)

    width = len(matrix[0]) if matrix else 0

    return height, width

