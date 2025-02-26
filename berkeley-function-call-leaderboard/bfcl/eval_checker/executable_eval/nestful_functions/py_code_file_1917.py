from typing import Dict



def generate_new_dict(old_dict: Dict[str, int]) -> Dict[str, str]:

    """Generates a new dictionary from the given dictionary, where the keys of the new dictionary are the values of the given dictionary, and the values are the keys, modified by only keeping the first letter and adding an asterisk (*) to the end.



    Args:

        old_dict: The original dictionary.



    Returns:

        A new dictionary.

    """

    new_dict = {}

    for key, value in old_dict.items():

        new_key = key[0] + '*'

        new_dict[new_key] = value

    return new_dict

