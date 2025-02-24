from typing import Tuple



def calculate_final_distance(x: float, y: float, z: float, t: float) -> float:

    """Calculates the final distance using the formula:

    D = sqrt((x + y)^2 + (z * t)^3)



    Args:

        x: The first value.

        y: The second value.

        z: The third value.

        t: The fourth value.

    """

    final_distance = ((x + y)**2 + (z * t)**3)**(1/2)

    return final_distance

