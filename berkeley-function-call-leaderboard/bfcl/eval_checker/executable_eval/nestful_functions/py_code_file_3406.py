from typing import Union



def convert_miles_to_kilometers(miles: Union[int, float]) -> float:

    """Converts a distance in miles to kilometers.



    Args:

        miles: The distance in miles.



    Returns:

        The equivalent distance in kilometers.

    """

    kilometers = miles * 1.60934  # Multiply miles by 1.60934 to convert them to kilometers

    return kilometers

