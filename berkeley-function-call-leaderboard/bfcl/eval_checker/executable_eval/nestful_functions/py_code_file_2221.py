from typing import Union



def kph_to_mph(kph: float) -> Union[float, str]:

    """

    Converts speed in kilometers per hour to miles per hour.

    Args:

        kph: The speed in kilometers per hour.

    """

    if kph <= 0:

        return "Invalid input: speed must be greater than zero."

    else:

        return kph * 0.621371

