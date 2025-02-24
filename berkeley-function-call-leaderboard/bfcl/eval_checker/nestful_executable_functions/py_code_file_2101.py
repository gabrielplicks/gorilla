from typing import List



def vectors_intersect(vector1: List[int], vector2: List[int]) -> bool:

    """Determines whether two vectors intersect.



    Args:

        vector1: The first vector.

        vector2: The second vector.



    Returns:

        True if the vectors intersect, False otherwise.

    """

    for element in vector1:

        if element in vector2:

            return True

    return False

