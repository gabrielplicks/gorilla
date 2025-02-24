from typing import List, Tuple, Optional



def find_first_empty_cell(grid: List[List[int]]) -> Optional[Tuple[int, int]]:

    """Finds the first empty cell position in a Sudoku grid.



    Args:

        grid: A 9x9 two-dimensional list of integers representing a Sudoku grid.



    Returns:

        The first empty cell position in the format `(x, y)` or `None` if there is no empty cell.

    """

    for row in range(len(grid)):

        for column in range(len(grid[row])):

            if grid[row][column] == 0:

                return (row, column)

    return None

