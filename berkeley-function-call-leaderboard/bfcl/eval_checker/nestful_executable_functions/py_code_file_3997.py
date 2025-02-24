from typing import Union



def kelvin_to_fahrenheit(temperature: Union[int, float]) -> int:

    """Convert a temperature from Kelvin to Fahrenheit.



    Args:

        temperature: The temperature in Kelvin.



    Returns:

        The temperature in Fahrenheit as an integer.

    """

    celsius = temperature - 273.15

    fahrenheit = (celsius * 1.8) + 32

    return int(fahrenheit)

