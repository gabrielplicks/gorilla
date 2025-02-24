from typing import Dict



def extract_even_pairs(original_dict: Dict[str, int]) -> Dict[str, int]:

    """Extracts key-value pairs from a dictionary where the value is even and builds a new dictionary from them.



    Args:

        original_dict: The input dictionary.



    Returns:

        A new dictionary containing only the key-value pairs where the value is even.

    """

    new_dict = {}



    for key, value in original_dict.items():

        if value % 2 == 0:

            new_dict[key] = value



    return new_dict

