import math

from typing import Union



def compute_area_circle(radius: Union[int, float]) -> float:

    """Computes the area of a circle given its radius.



    Args:

        radius: The radius of the circle.



    Raises:

        TypeError: If the input is not a number.

        ValueError: If the input is negative.

    """

    if not isinstance(radius, (int, float)):

        raise TypeError("Invalid input type. Expected a number.")

    if radius < 0:

        raise ValueError("Negative radius not allowed.")



    area = math.pi * (radius ** 2)

    return round(area, 2)

