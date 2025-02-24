from typing import Union



def calculate_square_area(l: Union[int, float]) -> int:

    """Calculates the area of a square given its length.



    Args:

        l: The length of the square.



    Raises:

        ValueError: If the provided value is not a positive integer.

    """

    if not isinstance(l, int) or l <= 0:

        raise ValueError("Length must be a positive integer.")

    return l * l

