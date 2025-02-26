from typing import Dict



def get_latitude(location: Dict[str, float]) -> float:

    """

    Returns the latitude value from a dictionary representing a GPS location.

    Args:

        location: A dictionary containing GPS location data.

    Raises:

        ValueError: If the latitude value is not found in the dictionary.

        TypeError: If the latitude value is not a number.

        ValueError: If the latitude value is not within the range of -90 to 90.

    """

    latitude = location.get('latitude', None)



    if latitude is None:

        raise ValueError('Latitude not found in the location dictionary.')

    elif not isinstance(latitude, (int, float)):

        raise TypeError('Latitude must be a number.')

    elif latitude > 90 or latitude < -90:

        raise ValueError('Latitude must be within the range of -90 to 90.')



    return latitude

