from typing import Dict, List



def get_keys_with_same_value(dictionary: Dict[str, int], value: int) -> List[str]:

    """Returns a list of keys from the dictionary that have the specified value.



    Args:

        dictionary: The dictionary to search.

        value: The value to search for.

    """

    return [key for key, v in dictionary.items() if v == value]

