from typing import Dict



def swap_dictionary(dictionary: Dict[int, str]) -> Dict[str, int]:

    """

    Swap the keys and values in a dictionary. For example,

    given `dictionary = {1: "a", 2: "b", 3: "c"}`, this function returns

    a new dictionary where the keys are swapped with their corresponding values.

    Args:

        dictionary: A dictionary with integer keys and string values.

    """

    swapped_dictionary = {}

    for key, value in dictionary.items():

        swapped_dictionary[value] = key

    return swapped_dictionary

