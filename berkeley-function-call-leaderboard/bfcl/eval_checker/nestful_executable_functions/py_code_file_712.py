from typing import List, Tuple



def largest_row_sum_index(matrix: List[List[float]]) -> int:

    """Returns the index of the row that contains the largest sum of its elements.

    If there is more than one row that satisfies this condition, returns the index of the first one.

    Args:

        matrix: A 2D array representing a matrix.

    """

    row_sums = {}

    for i, row in enumerate(matrix):

        row_sums[i] = sum(row)



    max_sum = max(row_sums.values())

    return [index for index, sum in row_sums.items() if sum == max_sum][0]

