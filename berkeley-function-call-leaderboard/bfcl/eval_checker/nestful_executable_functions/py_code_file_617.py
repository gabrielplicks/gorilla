from typing import Union



def temp_converter(value: Union[int, float], unit: str) -> Union[int, float]:

    """Converts a temperature value from celsius to fahrenheit and vice versa.



    Args:

        value: The temperature value to be converted.

        unit: The unit of the temperature value. It can be either 'C' or 'F'.



    Returns:

        The converted temperature value.

    """

    if unit == 'C':

        return (9/5) * value + 32

    elif unit == 'F':

        return (5/9) * (value - 32)

    else:

        raise ValueError('Unit must be either "C" or "F".')

