from typing import List, Dict



def dict_from_keys_recursive(keys: List[str], index: int = 0) -> Dict[str, str]:

    """

    Creates a dictionary from the given list of keys, where the keys are also used as the values.

    Args:

        keys: A list of keys to be used as both keys and values in the dictionary.

        index: The index of the current key in the list.

    """

    if index >= len(keys):

        return {}

    return {keys[index]: keys[index], **dict_from_keys_recursive(keys, index + 1)}

