from typing import List, Tuple



def linearize_matrix_index_pairs(matrix: List[List[int]], index_pairs: List[Tuple[int, int]]) -> List[int]:

    """Transforms a set of 2D indices into a list of linearized indices.



    Args:

        matrix: The matrix to be linearized.

        index_pairs: A list of 2D indices.



    Returns:

        A list of linearized indices.

    """

    rows = len(matrix)

    cols = len(matrix[0])

    linearized = []

    for row, col in index_pairs:

        linearized_index = row * cols + col

        linearized.append(linearized_index)

    return linearized

