import math



def convert_angle(a: float, unit_from: str) -> tuple:

    """Converts a given angle from one unit of measurement to another.



    Args:

        a: The given angle.

        unit_from: The current unit of measurement.



    Raises:

        ValueError: If the unit_from is not 'degrees' or 'radians'.

    """

    if unit_from == 'degrees':

        angle_radians = math.radians(a)

        return angle_radians, 'radians'

    elif unit_from == 'radians':

        angle_degrees = math.degrees(a)

        return angle_degrees, 'degrees'

    else:

        raise ValueError('Unit must be "degrees" or "radians"')

