from typing import Tuple



def grid_location_to_index(x: int, y: int) -> int:

    """Converts a 2D location on a 10 x 10 grid to a 1D index.



    The resulting index ranges from 0 to 99. The 1D index represents the x and y

    coordinates of the 2D location, where x is the horizontal axis and y is the

    vertical axis.



    Args:

        x: The x coordinate of the location.

        y: The y coordinate of the location.



    Raises:

        ValueError: If the location is out of bounds.

    """

    if x < 0 or x >= 10 or y < 0 or y >= 10:

        raise ValueError("Location out of bounds")



    return y * 10 + x

