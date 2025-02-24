from typing import Union



def convert_to_cm_with_precision(inches: Union[int, float]) -> float:

    """Converts a distance from inches to centimeters with a precision of 2 decimal places.



    Args:

        inches: The distance in inches.



    Returns:

        The distance in centimeters with a precision of 2 decimal places.

    """

    cm = inches * 2.54  # Convert to centimeters

    return round(cm, 2)  # Round to 2 decimal places

