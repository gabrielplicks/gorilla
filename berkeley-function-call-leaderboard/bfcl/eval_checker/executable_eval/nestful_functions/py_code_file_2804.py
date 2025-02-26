from typing import Union



def format_temperature(temperature_f: Union[int, float]) -> str:

    """Formats and returns the current temperature in degrees Celsius.



    Args:

        temperature_f: The temperature in degrees Fahrenheit.



    Returns:

        A string in the format "It is currently xx degrees Celsius.", where `xx` is the temperature in degrees Celsius.

    """

    temperature_c = (temperature_f - 32) * 5 / 9

    return "It is currently {:.1f} degrees Celsius.".format(temperature_c)

