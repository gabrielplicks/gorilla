import math



def slope_from_angle(angle: float) -> float:

    """Calculates the slope of a line given its angle in degrees.



    Args:

        angle: The angle in degrees between 0 and 90.



    Returns:

        The slope of the line that forms an angle of this value with the positive x-axis.

    """

    angle_radians = math.radians(angle)

    slope = math.tan(angle_radians)

    return float(slope)

