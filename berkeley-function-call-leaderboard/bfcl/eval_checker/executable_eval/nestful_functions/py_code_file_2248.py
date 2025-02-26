from typing import Tuple



def normalise_coordinates(x: float, y: float) -> Tuple[float, float]:

    """Normalises the x and y coordinates of a 2D point.



    Args:

        x: The x-coordinate of the point.

        y: The y-coordinate of the point.



    Returns:

        The normalised x and y values.

    """

    total = x + y

    return x / total, y / total

