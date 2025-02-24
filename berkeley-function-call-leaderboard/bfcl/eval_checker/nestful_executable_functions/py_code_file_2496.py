from typing import Dict



def get_new_dict(input_dict: Dict[str, str]) -> Dict[str, str]:

    """Returns a new dictionary with all keys in lowercase and any keys starting with "id_" removed.



    Args:

        input_dict: The input dictionary.



    Returns:

        A new dictionary with the specified modifications.

    """

    lowercase_dict = {key.lower(): value for key, value in input_dict.items()}

    return {key: value for key, value in lowercase_dict.items() if not key.startswith("id_")}

