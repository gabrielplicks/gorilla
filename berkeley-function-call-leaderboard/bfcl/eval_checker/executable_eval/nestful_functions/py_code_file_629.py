from typing import Dict



def remove_function_closures(data: Dict[str, any]) -> Dict[str, any]:

    """Removes all key-value pairs from the dictionary where the value is a function closure.

    Args:

        data: The dictionary to filter.

    Returns:

        A new dictionary with the filtered key-value pairs.

    """

    new_dictionary = {}

    for key, value in data.items():

        if not callable(value):

            new_dictionary[key] = value

    return new_dictionary

