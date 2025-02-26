from typing import List



def remove_row_col(matrix: List[List[int]], row_index: int, col_index: int) -> List[List[int]]:

    """Removes a specified row and column from a given matrix.

    Args:

        matrix: The matrix represented by a 2D list.

        row_index: The row number to be removed.

        col_index: The column number to be removed.

    """

    new_matrix = [row for i, row in enumerate(matrix) if i != row_index]



    if new_matrix:

        new_matrix = [

            [col for j, col in enumerate(row) if j != col_index]

            for row in new_matrix

        ]



    return new_matrix

