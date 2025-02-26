from typing import Union



def calculate_density(mass: Union[int, float], volume: Union[int, float]) -> float:

    """Calculates the density of a material based on the given formula: density = mass / volume.



    Args:

        mass: The mass of the material.

        volume: The volume of the material.



    Returns:

        The density of the material.

    """

    mass = float(mass)

    volume = float(volume)



    density = mass / volume



    return density

