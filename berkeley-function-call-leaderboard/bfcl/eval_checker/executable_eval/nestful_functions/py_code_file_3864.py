from typing import Dict



def find_highest_value_key(dictionary: Dict[str, int]) -> str:

    """Finds the key with the highest value in a dictionary using list comprehension.



    Args:

        dictionary: The dictionary to search.



    Returns:

        The key with the highest value.

    """

    return max(dictionary, key=lambda k: dictionary[k])

