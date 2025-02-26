from typing import Union



def speed_conversion(speed: float, unit: Union[str, None]) -> float:

    """Converts speed values from the following units to `m/s`: `km/h`, `mph`, `m/s` (already in `m/s`).

    Args:

        speed: The speed value to be converted.

        unit: The unit of the speed value.

    """

    if unit == 'km/h':

        return speed * 0.2777777777777778

    elif unit == 'mph':

        return speed * 0.44704

    elif unit == 'm/s':

        return speed

    else:

        raise ValueError("Invalid unit provided: {}".format(unit))

