from typing import Dict



def set_to_none(dictionary: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, None]]:

    """

    Returns a new dictionary with the same structure as the input dictionary, but with all the values set to None.

    The function can handle nested dictionaries and makes deep copies of all the values.



    Args:

        dictionary: The input dictionary.



    Returns:

        A new dictionary with the same structure as the input dictionary, but with all the values set to None.

    """

    dictionary_copy = {}



    for key, value in dictionary.items():

        if isinstance(value, dict):

            dictionary_copy[key] = set_to_none(value)

        else:

            dictionary_copy[key] = None



    return dictionary_copy

