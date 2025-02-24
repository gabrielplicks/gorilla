from typing import Tuple



def area_rectangle(length: float, width: float) -> float:

    """Calculates the area of a rectangle given the length and width.

    Args:

        length: The length of the rectangle.

        width: The width of the rectangle.

    """

    return length * width



assert area_rectangle(1, 1) == 1

assert area_rectangle(4, 5) == 20

assert area_rectangle(2, 3) == 6

