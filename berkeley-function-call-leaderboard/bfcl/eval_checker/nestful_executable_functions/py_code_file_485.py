from typing import Dict, List



def sort_keys_by_value(dictionary: Dict[str, int]) -> List[str]:

    """Sorts the keys of a dictionary by their corresponding values in descending order.



    Args:

        dictionary: The dictionary to sort.



    Returns:

        A list of keys sorted by their corresponding values in descending order.

    """

    return [key for key, _ in sorted(dictionary.items(), key=lambda x: x[1], reverse=True)]

