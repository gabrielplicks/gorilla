from typing import List, Tuple



def matrix_to_list_of_tuples(matrix: List[List[int]]) -> List[Tuple[int, int, int]]:

    """Converts a matrix (list of lists) into a list of tuples, where each tuple contains the element and its corresponding row and column indices.



    Args:

        matrix: The input matrix as a list of lists.



    Returns:

        A list of tuples, where each tuple contains the element and its corresponding row and column indices.

    """

    result = []

    for row_index, row in enumerate(matrix):

        for col_index, element in enumerate(row):

            result.append((element, row_index, col_index))

    return result

