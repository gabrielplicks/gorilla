from typing import Tuple



def kelvin_to_fahrenheit_celsius(temperature: float) -> Tuple[float, float]:

    """Converts a temperature in degrees Kelvin to degrees Fahrenheit and Celsius.



    Args:

        temperature: The temperature in Kelvin.



    Returns:

        A tuple of two values: the first one is the temperature in degrees Fahrenheit,

        and the second one is the temperature in degrees Celsius.

    """

    assert temperature > 0, "Temperature must be greater than 0 Kelvin"

    fahrenheit = (9/5) * (temperature - 273.15) + 32

    celsius = temperature - 273.15

    return fahrenheit, celsius

