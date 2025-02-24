import math



def convert_to_spherical(coordinates: list) -> list:

    """Converts a list of cartesian coordinates to spherical coordinates.



    Args:

        coordinates: A list of cartesian coordinates in the format [x, y, z].



    Returns:

        A list of spherical coordinates in the format [r, theta, phi].

    """

    x, y, z = coordinates

    r = math.sqrt(x**2 + y**2 + z**2)

    theta = math.atan2(y, x)

    phi = math.atan2(z, math.sqrt(x**2 + y**2))

    return [r, theta, phi]

