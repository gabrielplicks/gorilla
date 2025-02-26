from typing import Dict



def select_pairs(dictionary: Dict[str, int]) -> Dict[str, int]:

    """Selects the key-value pairs from a dictionary where the values are greater than or equal to 10.



    Args:

        dictionary: The input dictionary.



    Returns:

        A new dictionary containing only the selected key-value pairs.

    """

    result = {}

    for key, value in dictionary.items():

        if value >= 10:

            result[key] = value

    return result

