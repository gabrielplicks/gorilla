from typing import Dict, Any



def key_values_swap(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:

    """Swaps the keys and values of a dictionary.



    Args:

        dictionary: The dictionary to swap keys and values.



    Returns:

        A new dictionary with keys and values swapped.

    """

    new_dictionary = {}

    for key, value in dictionary.items():

        new_dictionary[value] = key

    return new_dictionary

