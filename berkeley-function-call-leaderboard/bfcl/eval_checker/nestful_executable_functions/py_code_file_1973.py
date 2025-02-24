from typing import List, Tuple



def retrieve_elements_below_diagonal(matrix: List[List[int]]) -> List[int]:

    """Retrieves the elements below the main diagonal of a square matrix.



    Args:

        matrix: A square matrix represented as a list of lists.



    Returns:

        A list of elements below the main diagonal.

    """

    elements = []

    for i in range(len(matrix)):

        for j in range(len(matrix[i])):

            if i > j:

                elements.append(matrix[i][j])

    return elements

