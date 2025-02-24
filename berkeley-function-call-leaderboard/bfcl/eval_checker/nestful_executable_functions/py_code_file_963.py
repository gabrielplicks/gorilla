from typing import Dict



def reverse_relationships(original_dict: Dict[str, str]) -> Dict[str, str]:

    """Reverses the relationships in a dictionary.



    Args:

        original_dict: The original dictionary.



    Returns:

        A new dictionary with the relationships reversed.

    """

    reversed_dict = {}

    for key, value in original_dict.items():

        reversed_dict[value] = key

    return reversed_dict

