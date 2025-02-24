from typing import Union



def convert_to_miles(distance: str) -> float:

    """Converts a distance in miles or kilometers to miles.



    Args:

        distance: A string representing a distance in miles or kilometers.



    Returns:

        The distance in miles.



    Raises:

        ValueError: If the input string does not contain a valid unit.

    """

    num, unit = distance.split()  # Separate the numerical part and the unit

    if unit == "km":

        return float(num) * 0.621371  # Convert kilometers to miles

    elif unit == "mi":

        return float(num)  # Return the numerical part as miles

    else:

        raise ValueError("Invalid unit provided")

