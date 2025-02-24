from typing import Union



def f_to_c(f: Union[int, float]) -> float:

    """Converts Fahrenheit to Celsius.



    Args:

        f: The temperature in Fahrenheit.



    Returns:

        The temperature in Celsius.

    """

    return (f - 32) * 5 / 9

