from typing import Union



def kilometers_to_miles(kilometers: Union[int, float]) -> float:

    """Converts a distance from kilometers to miles.

    Args:

        kilometers: The distance in kilometers.

    """

    miles = kilometers * 0.6213712  # Conversion factor

    return round(miles, 2)  # Round to two decimal places

