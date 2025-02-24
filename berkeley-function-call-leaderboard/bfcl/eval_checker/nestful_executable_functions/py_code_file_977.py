from typing import List



def dict_from_two_lists(keys: List[str], values: List[str]) -> dict:

    """Creates a dictionary from two lists, where keys[i] is the key for values[i] in the result.

    If keys and values are not the same length, then you should ignore the excess elements beyond the length of the shorter list.

    Args:

        keys: A list of keys.

        values: A list of values.

    """

    min_length = min(len(keys), len(values))

    return {key: value for key, value in zip(keys[:min_length], values[:min_length])}

