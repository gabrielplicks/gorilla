import numpy as np



def get_volume_of_sphere_from_diameter(diameter: float) -> np.float64:

    """Calculates the volume of a sphere given its diameter.



    Args:

        diameter: The diameter of the sphere.



    Returns:

        The volume of the sphere as a float.

    """

    radius = diameter / 2

    volume = 4/3 * np.pi * radius**3

    return np.float64(volume)

