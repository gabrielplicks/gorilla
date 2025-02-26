from typing import Union

from math import pi



def degree_to_radian(degree: Union[int, float]) -> float:

    """Converts a degree value to a radian value, where the output should be between -180 and 180 degrees.

    The function is self-contained and does not rely on any external libraries.



    Args:

        degree: The degree value to be converted to radian.

    """

    radian = degree * (pi / 180)

    radian %= 360

    if radian > 180:

        radian -= 360



    return radian

