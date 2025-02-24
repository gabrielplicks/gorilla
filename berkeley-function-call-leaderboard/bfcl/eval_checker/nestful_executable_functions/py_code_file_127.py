from typing import Tuple



def get_grid_cell_corners(centre_x: float, centre_y: float, cell_size: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:

    """Calculates the lower-left and upper-right corners of a grid cell given its centre coordinates and cell size.

    Args:

        centre_x: The x-coordinate of the centre of the grid cell.

        centre_y: The y-coordinate of the centre of the grid cell.

        cell_size: The size of the grid cell.

    """

    lower_left_x = centre_x - cell_size / 2

    lower_left_y = centre_y - cell_size / 2

    upper_right_x = centre_x + cell_size / 2

    upper_right_y = centre_y + cell_size / 2

    return (lower_left_x, lower_left_y), (upper_right_x, upper_right_y)

