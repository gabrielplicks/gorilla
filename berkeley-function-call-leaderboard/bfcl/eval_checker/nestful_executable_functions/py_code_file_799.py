from typing import List, Union



def get_matrix_element(matrix: List[List[Union[int, float]]], row: int, column: int) -> Union[int, float]:

    """Returns the element of the specified row and column in the matrix.



    Args:

        matrix: The matrix to search.

        row: The row index of the desired element.

        column: The column index of the desired element.



    Raises:

        ValueError: If the matrix has an invalid shape.

        IndexError: If the row and/or column indices are out of bounds.

    """

    rows = len(matrix)

    columns = len(matrix[0]) if rows > 0 else 0

    if rows <= 0 or columns <= 0:

        raise ValueError("The matrix has an invalid shape")

    if row < 0 or row >= rows or column < 0 or column >= columns:

        raise IndexError("The row and/or column indices are out of bounds")



    return matrix[row][column]

