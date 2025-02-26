from typing import Tuple



def distance_from_plane(point: Tuple[float, float, float], plane: Tuple[float, float, float]) -> float:

    """Calculates the distance of a 3D point from a plane.



    Args:

        point: The 3D point represented as a tuple of three numbers.

        plane: The 3D plane represented by a tuple of three numbers, where the numbers form the coefficients of the plane equation `ax + by + cz = 0`.



    Returns:

        The distance between the point and the plane as a float.

    """

    x, y, z = point

    a, b, c = plane

    distance = abs(a * x + b * y + c * z) / (a**2 + b**2 + c**2)**0.5

    return distance

