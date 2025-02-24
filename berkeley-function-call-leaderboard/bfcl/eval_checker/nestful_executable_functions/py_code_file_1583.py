from typing import List, Tuple, Union



def get_value(matrix: List[List[Union[int, float]]], indices: Tuple[int, int]) -> Union[int, float, None]:

    """Gets the value at the specified index in the matrix.



    Args:

        matrix: The matrix represented as a list of lists.

        indices: The indices of the value to retrieve.



    Returns:

        The value at the specified index in the matrix, or None if the indices are out of range.

    """

    row_index, col_index = indices

    if row_index < 0 or row_index >= len(matrix) or \

        col_index < 0 or col_index >= len(matrix[row_index]):

        return None

    return matrix[row_index][col_index]

