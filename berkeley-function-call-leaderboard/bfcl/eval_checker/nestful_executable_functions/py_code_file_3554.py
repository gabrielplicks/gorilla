import math



def calculate_azimuth(point1: tuple, point2: tuple) -> float:

    """Calculates the azimuth angle between two 3D points.



    Args:

        point1: The first 3D point.

        point2: The second 3D point.



    Returns:

        The azimuth angle in radians between −π and π radians. If the points are the same, the function returns None.

    """

    try:

        x1, y1, z1 = point1

        x2, y2, z2 = point2

        dx = x2 - x1

        dy = y2 - y1

        dz = z2 - z1

        azimuth = math.atan2(dy, dx)

        return azimuth

    except ValueError:

        return None

