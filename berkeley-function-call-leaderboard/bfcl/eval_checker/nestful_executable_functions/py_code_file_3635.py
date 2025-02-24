from typing import Dict



def sort_keys(input_dict: Dict[str, any]) -> Dict[str, any]:

    """Sorts the keys of a dictionary alphabetically and returns a new dictionary.

    Args:

        input_dict: The input dictionary.

    Raises:

        TypeError: If the input dictionary contains any non-string keys.

    """

    if not all(isinstance(key, str) for key in input_dict.keys()):

        raise TypeError("Non-string keys found in the input dictionary.")

    sorted_keys = sorted(input_dict.keys())

    return {key: input_dict[key] for key in sorted_keys}

