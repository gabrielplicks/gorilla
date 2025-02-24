from typing import Union



def calculate_elastic_modulus(E_Y: Union[int, float], v: Union[int, float]) -> Union[int, float]:

    """Calculates the elastic modulus of a material based on its Young's modulus and Poisson's ratio.



    Args:

        E_Y: The Young's modulus of the material.

        v: The Poisson's ratio of the material.



    Raises:

        ValueError: If the Poisson's ratio is outside the range [0, 1].

    """

    if not 0 <= v <= 1:

        raise ValueError("poisson ratio must be in range [0, 1]")

    return E_Y / (1 - v ** 2)

