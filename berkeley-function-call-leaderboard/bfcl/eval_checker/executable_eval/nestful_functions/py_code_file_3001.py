from typing import Union



def cube_density(length: Union[int, float], density: Union[int, float]) -> Union[int, float]:

    """Calculates the density of a cube given its dimensions and density.



    Args:

        length: The length of each edge of the cube.

        density: The density of the cube.



    Returns:

        The density of the cube.

    """

    volume = length ** 3

    mass = density * volume

    return mass / volume

