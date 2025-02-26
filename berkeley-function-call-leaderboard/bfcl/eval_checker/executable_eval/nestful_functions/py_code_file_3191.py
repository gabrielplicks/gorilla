from typing import Union



def validate_geo_coordinates(latitude: Union[int, float], longitude: Union[int, float]) -> bool:

    """Validates a set of geographical coordinates (latitude and longitude).



    Args:

        latitude: The latitude of the geographical coordinate.

        longitude: The longitude of the geographical coordinate.



    Returns:

        True if the coordinates are valid, False otherwise.

    """

    if -90 <= latitude <= 90 and -180 <= longitude <= 180:

        return True

    else:

        return False

