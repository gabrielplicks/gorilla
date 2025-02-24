from typing import Dict



def validate_and_update(dictionary: Dict, key: str) -> Dict:

    """Validates and updates a dictionary with a given key.



    Args:

        dictionary: The input dictionary.

        key: The key to be added or updated in the dictionary.



    Returns:

        The updated dictionary.

    """

    if not dictionary:

        return {}

    if key in dictionary:

        dictionary[key] += 1

    else:

        dictionary[key] = 1



    return dictionary

