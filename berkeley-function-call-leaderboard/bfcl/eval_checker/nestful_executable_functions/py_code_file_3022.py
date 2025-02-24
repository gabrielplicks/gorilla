from typing import Union



def compute_temperature(temperature_in_fahrenheit: Union[int, float]) -> float:

    """Computes the temperature in Celsius from a given Fahrenheit value.



    Args:

        temperature_in_fahrenheit: The temperature in Fahrenheit.



    Returns:

        The temperature in Celsius.

    """

    temperature_in_celsius = (temperature_in_fahrenheit - 32) * 5 / 9

    return temperature_in_celsius

