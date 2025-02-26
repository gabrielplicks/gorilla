from typing import Dict



def convert_keys_to_str(dictionary: Dict[object, object]) -> Dict[str, object]:

    """Returns a copy of a given dictionary with all keys converted to strings.



    Args:

        dictionary: The input dictionary.



    Returns:

        A new dictionary with all keys converted to strings.

    """

    result = {}



    for key, value in dictionary.items():

        result[str(key)] = value



    return result

