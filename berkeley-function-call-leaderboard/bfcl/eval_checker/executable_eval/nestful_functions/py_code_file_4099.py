from typing import Union



def calculate_concentration(mass: Union[int, float], volume: Union[int, float]) -> Union[int, float]:

    """Calculates the concentration of a protein sample given its mass and volume.



    Args:

        mass: The mass of the protein sample.

        volume: The volume of the protein sample.



    Returns:

        The concentration of the protein sample in units of mass per unit volume.

    """

    concentration = mass / volume

    return concentration

