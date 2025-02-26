from typing import Dict



def get_default_dict() -> Dict[str, Dict[str, str]]:

    """Returns a dictionary with a specific structure for storing information in a caching system.



    Args:

        None



    Returns:

        A dictionary with the following structure:

        {

            "client": {},

            "rooms": {},

            "houses": {},

            "entities": {},

            "relationships": {},

        }

    """

    keys = ["client", "rooms", "houses", "entities", "relationships"]

    return {key: {} for key in keys}

