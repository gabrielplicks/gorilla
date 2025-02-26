from typing import Dict



def map_to_nested_dict(key_name: str, key_values: Dict[str, int]) -> Dict[str, Dict[str, int]]:

    """Maps a `key_name` to a dictionary of `key_values` and returns a new dictionary with the `key_name` replaced by a nested dictionary of `key_values`.



    Args:

        key_name: The name of the key to be replaced by a nested dictionary.

        key_values: A dictionary of string keys and integer values.



    Returns:

        A new dictionary with the `key_name` mapped to a nested dictionary of `key_values`.

    """

    return {key_name: {k: v for k, v in key_values.items()}}  # Dict comprehension

