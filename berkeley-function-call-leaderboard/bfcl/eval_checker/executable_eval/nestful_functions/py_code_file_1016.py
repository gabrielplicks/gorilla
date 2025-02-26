from typing import Union



def convert_miles_km(miles: Union[int, float]) -> float:

    """Converts miles to kilometers using the formula `miles * 1.609344`.



    Args:

        miles: The distance in miles.



    Returns:

        The distance in kilometers.

    """

    return miles * 1.609344

