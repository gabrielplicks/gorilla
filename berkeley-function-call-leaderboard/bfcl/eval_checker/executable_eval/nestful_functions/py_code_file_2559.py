from typing import Dict



def replace_with_lengths(dictionary: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, int]]:

    """Replaces the values in a dictionary with their lengths.

    If the value is not a dictionary, its length is set to 1.

    Args:

        dictionary: The input dictionary.

    Returns:

        A dictionary with the same structure as the input dictionary, but with the values replaced by their lengths.

    """

    result = {}

    for key, value in dictionary.items():

        if isinstance(value, dict):

            result[key] = replace_with_lengths(value)

        else:

            result[key] = 1

    return result

