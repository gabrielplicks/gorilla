from typing import Dict



def update_or_add(dict1: Dict[str, any], dict2: Dict[str, any]) -> None:

    """Updates the values of the first dictionary if the key exists in both dictionaries, otherwise adds the new key-value pair to the first dictionary.



    Args:

        dict1: The first dictionary to be updated.

        dict2: The second dictionary containing the key-value pairs to be added or updated.

    """

    for key, value in dict2.items():

        if key in dict1:

            dict1[key] = value

        else:

            dict1[key] = value

