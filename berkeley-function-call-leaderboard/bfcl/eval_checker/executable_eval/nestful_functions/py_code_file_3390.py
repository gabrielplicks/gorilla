from typing import List



def create_grid(rows: int, cols: int) -> List[List[int]]:

    """Creates a 2D matrix of size rows x cols filled with zeros.



    Args:

        rows: The number of rows in the matrix.

        cols: The number of columns in the matrix.

    """

    grid = []

    for row in range(rows):

        grid.append([])

        for col in range(cols):

            grid[row].append(0)

    return grid

