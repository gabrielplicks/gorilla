from typing import List



def traverse_diagonal(matrix: List[List[int]]) -> List[int]:

    """Traverses a square matrix and returns the diagonal elements, ordered from the top-left to the bottom-right.



    Args:

        matrix: A square matrix represented as a list of lists.



    Returns:

        A list of the diagonal elements.

    """

    diagonal_elements = []

    num_rows = len(matrix)



    for i in range(num_rows):

        diagonal_elements.append(matrix[i][i])



    return diagonal_elements

