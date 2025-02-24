from typing import Dict, List



def get_values_greater_than_zero(dictionary: Dict[str, int], keys: List[str]) -> List[int]:

    """Returns a list of all the values from the dictionary that correspond to the keys in the list.

    Only includes these values if the values are greater than 0.

    If there are no corresponding values, then returns an empty list.



    Args:

        dictionary: A dictionary whose keys are strings and values are integers.

        keys: A list of keys.

    """

    return [value for key, value in dictionary.items() if key in keys and value > 0]

