from typing import List



def column_sum(matrix: List[List[int]]) -> List[int]:

    """Calculates the sum of each column in a matrix.



    Args:

        matrix: A matrix represented as a list of lists.



    Returns:

        A list of the sum of each column in the matrix.

    """

    column_sums = []

    for column_index, column in enumerate(matrix[0]):

        column_sum = 0

        for row_index, row in enumerate(matrix):

            column_sum += row[column_index]

        column_sums.append(column_sum)

    return column_sums

