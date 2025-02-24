import math



def convert_radian_to_degree(radian: float) -> float:

    """Converts radians to degrees.



    Args:

        radian: The value in radians to be converted to degrees.

    """

    degree = radian * 180 / math.pi

    return degree

