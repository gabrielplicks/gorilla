from typing import List



def get_sparse_matrix_unique_values(sparse_matrix: List[List[int]]) -> List[int]:

    """

    Returns a sorted list containing all unique counter values from the sparse matrix.

    A counter value is the value of a cell in the matrix.



    Args:

        sparse_matrix: A two-dimensional list of lists containing non-negative integers.

    """

    flattened_list = sum(sparse_matrix, [])

    unique_values = set(flattened_list)

    sorted_unique_values = sorted(unique_values)

    return sorted_unique_values

