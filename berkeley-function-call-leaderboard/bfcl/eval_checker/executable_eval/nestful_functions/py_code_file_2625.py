from typing import Dict



def get_first_n_pairs(dictionary: Dict[str, int], n: int) -> Dict[str, int]:

    """Returns a new dictionary containing only the first `n` key-value pairs from the input dictionary.



    Args:

        dictionary: The input dictionary.

        n: The number of key-value pairs to include in the new dictionary.



    Returns:

        A new dictionary containing only the first `n` key-value pairs from the input dictionary.

    """

    new_dictionary = {}

    for key, value in dictionary.items():

        new_dictionary[key] = value

        if len(new_dictionary) == n:

            break



    return new_dictionary

