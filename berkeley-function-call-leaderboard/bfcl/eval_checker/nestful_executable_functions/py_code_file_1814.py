from typing import Dict, List, Tuple



def max_key_value(dictionary: Dict[str, int]) -> Tuple[List[str], int]:

    """

    Returns a tuple containing the keys corresponding to the maximum value and the maximum value itself.

    If there are multiple keys with the same maximum value, they are all returned in a list.



    Args:

        dictionary: The input dictionary.

    """

    max_key_value = max(dictionary.values())  # Get the maximum value in the dictionary

    max_keys = [key for key, value in dictionary.items() if value == max_key_value]

    return max_keys, max_key_value

