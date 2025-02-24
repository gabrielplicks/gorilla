import math



def calculate_sphere_volume(radii: list[float]) -> list[float]:

    """Calculates the volume of a sphere with a given radius.



    Args:

        radii: A list of floats, each representing a radius.



    Raises:

        ValueError: If an input radius is less than or equal to 0.

    """

    volumes = []

    for r in radii:

        if r <= 0:

            raise ValueError("Radius must be greater than 0.")

        volumes.append(4/3 * math.pi * r**3)

    return volumes

