from typing import List, Tuple



def find_empty_spaces(grid: List[List[int]]) -> List[Tuple[int, int]]:

    """Finds all the empty spaces in a 2D grid.



    Args:

        grid: A 2D grid of 0's and 1's, where 0's represent empty spaces and 1's represent obstacles.



    Returns:

        A list of coordinates representing all the empty spaces in the grid.

    """

    return [(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] == 0]

