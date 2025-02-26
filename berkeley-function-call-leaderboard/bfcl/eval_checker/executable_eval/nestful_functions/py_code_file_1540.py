from typing import Tuple



def cell_coordinate(cell: int, num_cells_in_row: int) -> Tuple[int, int]:

    """Determines the coordinates of a cell in a two-dimensional grid based on the cell number and number of cells in each row.



    Args:

        cell: The cell number.

        num_cells_in_row: The number of cells in each row of the grid.



    Returns:

        A tuple `(row, col)` representing the coordinates of the cell in the grid.

    """

    row = (cell - 1) // num_cells_in_row + 1

    col = (cell - 1) % num_cells_in_row + 1

    return (row, col)

