from math import sqrt, atan2

from typing import Tuple



def convert_cartesian_to_cylindrical(x: float, y: float, z: float) -> Tuple[float, float, float]:

    """Converts a Cartesian coordinate to a cylindrical coordinate.



    Args:

        x: The x-coordinate of the Cartesian coordinate.

        y: The y-coordinate of the Cartesian coordinate.

        z: The z-coordinate of the Cartesian coordinate.



    Returns:

        A tuple containing the cylindrical coordinates (r, theta, z).

    """

    r = sqrt(x**2 + y**2)

    theta = atan2(y, x)

    return r, theta, z

