from typing import Tuple



def compute_pm(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:

    """Calculates the proper motion of a star given its coordinates in different epochs.

    Args:

        x1: The x-coordinate of the star in the first epoch.

        y1: The y-coordinate of the star in the first epoch.

        x2: The x-coordinate of the star in the second epoch.

        y2: The y-coordinate of the star in the second epoch.

    Returns:

        A tuple containing the proper motion in the x and y directions.

    """

    dx = x2 - x1

    dy = y2 - y1

    return dx, dy

