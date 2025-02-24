from typing import Union



def is_valid_matrix_input(matrix: Union[int, float]) -> bool:

    """Checks if a number is a valid input for a matrix.



    Args:

        matrix: The number to check.



    Returns:

        True if the input is valid, False otherwise.

    """

    if not isinstance(matrix, (int, float)):

        return False

    if not matrix in range(-100, 101):

        return False

    return True

