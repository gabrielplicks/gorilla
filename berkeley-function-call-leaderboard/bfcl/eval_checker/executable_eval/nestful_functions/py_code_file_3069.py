from typing import Tuple



def check_overlap(range1: Tuple[float, float], range2: Tuple[float, float]) -> bool:

    """Checks if two frequency ranges overlap.



    Args:

        range1: The first frequency range.

        range2: The second frequency range.



    Returns:

        True if the ranges overlap, False otherwise.

    """

    if (range1[0] > range2[1]) or (range2[0] > range1[1]):

        return False

    else:

        return True

