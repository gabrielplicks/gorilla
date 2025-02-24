import logging

from typing import Dict, Union



def map_location_type(location_type: str, loc_types: Dict[str, str]) -> Union[str, None]:

    """Maps a location type value to a string name using the loc_types dictionary.



    Args:

        location_type: The location type value.

        loc_types: The dictionary mapping location type values to string names.



    Returns:

        The mapped string name if the location type value is valid and exists in the loc_types dictionary.

        None if the location type value is invalid or does not exist in the loc_types dictionary.

    """

    if location_type not in ["201", "202"]:

        logging.warning(f"Invalid location type: {location_type}")

        return None



    try:

        return loc_types[location_type]

    except KeyError:

        logging.warning(f"Unknown location type: {location_type}")

        return None

