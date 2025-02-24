from typing import Union



def convert_from_kmh_to_mph(kph: Union[int, float]) -> Union[float, None]:

    """Converts a value in kilometers per hour to miles per hour.



    Args:

        kph: The value in kilometers per hour.



    Returns:

        The value in miles per hour, rounded to four decimal places.

        If the input value is not a number, returns None and prints an error message.

    """

    try:

        mph = round(kph * 0.621371, 4)

        return mph

    except TypeError:

        print("Invalid input type for kph. Please provide a valid number.")

        return None

