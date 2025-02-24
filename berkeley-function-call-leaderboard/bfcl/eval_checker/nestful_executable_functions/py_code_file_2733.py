from typing import Union



def convert_voltage(value: Union[int, float], unit: str) -> Union[float, str]:

    """Converts a voltage from one unit to another.

    Args:

        value: The numerical value of the voltage.

        unit: The unit of the voltage.

    """

    valid_units = ['V', 'mV', 'µV']

    if unit not in valid_units:

        return f'Error: Invalid unit {unit}. Valid units are {", ".join(valid_units)}.'

    value = float(value)

    if unit == 'mV':

        value /= 1000

    elif unit == 'µV':

        value /= 1000000



    return value

