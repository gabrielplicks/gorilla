from typing import Dict



def update_and_remove(dict1: Dict[str, int], dict2: Dict[str, int]) -> Dict[str, int]:

    """Updates a dictionary with the contents of another dictionary and removes all the keys that are not in the provided dictionary.



    Args:

        dict1: The dictionary to be updated.

        dict2: The dictionary containing the keys to be kept.



    Returns:

        The updated dictionary.

    """

    dict1_copy = dict1.copy()

    for key in [key for key in dict1_copy.keys() if key not in dict2]:

        del dict1_copy[key]

    dict1_copy.update(dict2)

    return dict1_copy

