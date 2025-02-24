import math



def convert_length_meters_to_feet(meters: float) -> float:

    """Converts a length from meters to feet.



    Args:

        meters: The length in meters.



    Returns:

        The equivalent length in feet, rounded to two decimal places.

        Returns None if the input length is less than 0.

    """

    if meters < 0:

        return None



    feet = meters * 3.28084

    return round(feet, 2)

