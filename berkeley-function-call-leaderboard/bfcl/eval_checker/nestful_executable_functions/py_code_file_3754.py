from typing import Dict



def filter_pairs(pairs: Dict[str, int]) -> Dict[str, int]:

    """Filters a dictionary of key-value pairs to only include those where the key is a string with at least 10 characters and the value is an integer greater than 100.



    Args:

        pairs: The dictionary of key-value pairs to filter.



    Returns:

        A new dictionary containing only the key-value pairs that satisfy the condition.

    """

    filtered_pairs = {}

    for key, value in pairs.items():

        if isinstance(key, str) and len(key) >= 10 and isinstance(value, int) and value > 100:

            filtered_pairs[key] = value

    return filtered_pairs

