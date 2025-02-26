from typing import Union



def number_to_celsius(input: Union[str, int, float]) -> Union[str, float]:

    """Converts a number to its corresponding temperature in Celsius.

    Args:

        input: The input number, which can be a `float`, `int`, or `str`.

    Returns:

        The temperature in Celsius as a `float`, or an error message as a `str`.

    """

    try:

        number = float(input)

    except ValueError:

        return "Error: invalid input."



    if number < 0:

        return "Error: negative numbers are not supported."



    celsius = number * (9 / 5) + 32

    rounded_celsius = round(celsius, 2)



    return rounded_celsius

