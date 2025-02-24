from typing import Union



def calculate_diameter(radius: Union[int, float]) -> float:

    """

    Calculate the diameter of a circle given its radius.

    The function rounds the result to three decimal digits.

    Args:

        radius: The radius of the circle.

    """

    diameter = 2 * radius

    return round(diameter, 3)

