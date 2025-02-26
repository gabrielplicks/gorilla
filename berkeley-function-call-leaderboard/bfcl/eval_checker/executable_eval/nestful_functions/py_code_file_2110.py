from typing import Dict



def reverse_dict_values(d: Dict[str, int]) -> Dict[int, str]:

    """Reverses the keys and values of a dictionary.



    Args:

        d: The dictionary to reverse.



    Returns:

        A new dictionary with the keys and values reversed.

    """

    reversed_dict = {}

    for key, val in d.items():

        reversed_dict[val] = key

    return reversed_dict

