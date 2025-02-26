from typing import List



def pascals_triangle(n: int) -> List[List[int]]:

    """Returns the n-th term of the Pascal's triangle.



    Args:

        n: The term number.



    Returns:

        A 2D list of integers, where each sublist represents a row.

    """

    triangle = [[1]]

    for i in range(1, n):

        row = [1]

        for j in range(1, i):

            row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])

        row.append(1)

        triangle.append(row)

    return triangle[n - 1]

