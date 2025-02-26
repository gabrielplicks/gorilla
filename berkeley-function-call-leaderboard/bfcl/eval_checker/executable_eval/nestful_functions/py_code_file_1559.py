from typing import Tuple



def grid_to_xy(cell_index: int, grid_width: int) -> Tuple[int, int]:

    """Converts a grid location to an (x, y) coordinate in 2D space.

    Args:

        cell_index: The index of the cell in the grid.

        grid_width: The width of the grid.

    """

    x = cell_index % grid_width

    y = cell_index // grid_width

    return x, y

