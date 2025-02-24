from typing import Dict, List



def sort_keys_alphabetically(dictionary: Dict[str, any]) -> List[str]:

    """Sorts the keys of a dictionary in alphabetical order, where the keys are first filtered by their first letter being a lowercase letter.



    Args:

        dictionary: The dictionary to sort.



    Returns:

        A list of the sorted keys.

    """

    filtered_keys = [

        key for key in dictionary

        if key[0].islower()

    ]

    sorted_keys = sorted(filtered_keys)

    return sorted_keys

