from typing import Dict



def swap_keys_and_values(input_dict: Dict[str, int]) -> Dict[int, str]:

    """Swaps keys and values of a dictionary.



    Args:

        input_dict: The input dictionary.



    Returns:

        A new dictionary with the keys and values swapped.

    """

    swapped_dict = {}

    for key, value in input_dict.items():

        swapped_dict[value] = key

    return swapped_dict

