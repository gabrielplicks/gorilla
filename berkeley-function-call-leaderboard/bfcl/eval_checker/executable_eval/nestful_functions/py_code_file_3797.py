from typing import Dict



def add_if_exists(dictionary: Dict[str, int], key1: str, key2: str) -> Dict[str, int]:

    """Adds the value of the second key to the value of the first key in a dictionary.



    If the first key doesn't exist, it is added to the dictionary with the value of the second key.



    Args:

        dictionary: The dictionary to update.

        key1: The first key.

        key2: The second key.



    Returns:

        The updated dictionary.

    """

    if key1 in dictionary:

        dictionary[key1] += dictionary[key2]

    else:

        dictionary[key1] = dictionary[key2]

    return dictionary

