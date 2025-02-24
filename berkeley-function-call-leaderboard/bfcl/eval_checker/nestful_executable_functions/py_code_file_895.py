from typing import Tuple



def map_fractional_to_cartesian(x: float, y: float) -> Tuple[int, int]:

    """Maps a fractional coordinate (x, y) to a Cartesian coordinate (x', y').



    The Cartesian coordinate system has a bottom-left origin, where the fractional coordinate (0, 0) maps to the Cartesian coordinate (4, 4).

    The transformation applied is (x', y') = (4 - x, 4 - y).



    Args:

        x: The x-coordinate of the fractional coordinate.

        y: The y-coordinate of the fractional coordinate.



    Returns:

        A tuple of integers representing the Cartesian coordinate (x', y').

    """

    x_prime = 4 - x

    y_prime = 4 - y

    return (x_prime, y_prime)

