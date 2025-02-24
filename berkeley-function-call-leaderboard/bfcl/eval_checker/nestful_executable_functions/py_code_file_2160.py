from typing import Dict



def flatten_dictionary_helper(dictionary: Dict, prefix: str, accumulator: Dict):

    """Flattens a nested dictionary into a single dictionary with concatenated keys.



    Args:

        dictionary: The nested dictionary to flatten.

        prefix: The current prefix for the keys.

        accumulator: The accumulator dictionary to store the flattened keys and values.

    """

    for key, value in dictionary.items():

        if isinstance(value, dict):

            flatten_dictionary_helper(value, prefix + key + '_', accumulator)

        else:

            accumulator[prefix + key] = value

    return accumulator



def flatten_dictionary(dictionary: Dict):

    """Flattens a nested dictionary into a single dictionary with concatenated keys.



    Args:

        dictionary: The nested dictionary to flatten.

    """

    return flatten_dictionary_helper(dictionary, '', {})

