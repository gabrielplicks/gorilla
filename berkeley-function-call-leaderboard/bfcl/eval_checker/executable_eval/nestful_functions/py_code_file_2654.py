from typing import Union



def mass_diff(mass_1: Union[int, float], mass_2: Union[int, float]) -> float:

    """Calculates the difference in kilograms between two masses given in grams.

    Args:

        mass_1: The first mass in grams.

        mass_2: The second mass in grams.

    """

    diff = abs(mass_1 - mass_2)

    diff_kg = diff / 1000.0

    return round(diff_kg, 2)

