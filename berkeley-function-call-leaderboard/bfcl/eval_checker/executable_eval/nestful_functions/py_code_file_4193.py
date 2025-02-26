from typing import Union



def scale_temperature_C_to_F(C: Union[int, float]) -> str:

    """Converts a temperature in degrees Celsius to degrees Fahrenheit.



    Args:

        C: A float representing a temperature in Celsius.



    Returns:

        A string representing the temperature in Fahrenheit with two decimal places of precision.



    Raises:

        TypeError: If the input is not a number.

    """

    if not isinstance(C, (int, float)):

        raise TypeError("Input must be a number")

    F = C * 9 / 5 + 32

    return f"{F:.2f}"

