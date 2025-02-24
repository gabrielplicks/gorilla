from typing import Dict



def match_keys_and_values(first_dict: Dict, second_dict: Dict) -> Dict:

    """Returns a dictionary with the keys and values that match in both dictionaries.



    Args:

        first_dict: The first dictionary.

        second_dict: The second dictionary.



    Returns:

        A dictionary with the keys and values that match in both dictionaries.

    """

    return {key: value for key, value in first_dict.items()

            if key in second_dict and value == second_dict[key]}

