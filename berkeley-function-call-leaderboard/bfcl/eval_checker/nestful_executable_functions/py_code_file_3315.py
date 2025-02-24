from typing import List



def extract_matrix_elements(matrix: List[List[int]]) -> List[int]:

    """Extracts all the elements from a homogeneous matrix and combines them into a single list.

    Args:

        matrix: A 2D list representing the matrix.

    Returns:

        A 1D list containing all the elements in the matrix.

    """

    extracted_elements = []

    for row in matrix:

        for element in row:

            extracted_elements.append(element)

    return extracted_elements

