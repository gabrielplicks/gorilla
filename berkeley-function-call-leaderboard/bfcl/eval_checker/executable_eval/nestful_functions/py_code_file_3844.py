from typing import Union

import math



def cosine(angle_degrees: Union[int, float]) -> str:

    """Calculates the exact value of the cosine function for a given angle in degrees.

    Args:

        angle_degrees: The angle in degrees.

    """

    angle_radians = angle_degrees * (math.pi / 180)

    return str(math.sin(angle_radians + math.pi / 2))

