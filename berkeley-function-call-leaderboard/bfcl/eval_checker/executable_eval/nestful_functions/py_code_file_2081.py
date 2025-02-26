from typing import List, Tuple



def neighbors(matrix: List[List[int]], r: int, c: int) -> List[Tuple[int, int]]:

    """

    Returns a list of tuples of the coordinates of all the neighbors of (r,c)

    that are within the bounds of the matrix.



    Args:

        matrix: The sparse matrix.

        r: The row coordinate of the given coordinate.

        c: The column coordinate of the given coordinate.

    """

    neighbors = []



    for i in range(r-1, r+2):

        for j in range(c-1, c+2):

            if 0 <= i < len(matrix) and 0 <= j < len(matrix[0]) and (i, j) != (r, c):

                neighbors.append((i, j))



    return neighbors

