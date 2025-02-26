from bisect import bisect_left

from typing import List



def calc_abundance(isotopes: List[int], abundances: List[float], isotope: int) -> float:

    """Calculates the abundance of a specific isotope based on a given set of isotopes and their abundances.

    Args:

        isotopes: A list of isotopes sorted by mass number.

        abundances: A list of abundances corresponding to the isotopes.

        isotope: The isotope to calculate the abundance for.

    """

    index = bisect_left(isotopes, isotope)

    if index < len(isotopes) and isotopes[index] == isotope:

        return abundances[index]

    elif index == len(isotopes):

        return 1.0

    else:

        return 0.0

