import math



def polygon(r: int, n: int) -> float:

    """Calculates the side length of a regular polygon inscribed in a circle.



    Args:

        r: The radius of the circle.

        n: The number of sides of the polygon.



    Raises:

        ValueError: If the number of sides is less than 3.

    """

    if n < 3:

        raise ValueError("The number of sides must be at least 3.")

    return (2 * math.pi * r) / n

