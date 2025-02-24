from typing import List



def get_main_diagonal(matrix: List[List[float]]) -> List[float]:

    """Returns a new list of the elements on the main diagonal of a 2D list.



    The main diagonal of a matrix is the diagonal line of elements that connect from the top left to the bottom right of the matrix. In the provided 2D list, the diagonal elements are the elements in the i-th row and j-th column, where i = j.



    Args:

        matrix: A 2D list containing numbers.



    Returns:

        A new list containing the diagonal elements.

    """

    return [row[i] for i, row in enumerate(matrix)]

