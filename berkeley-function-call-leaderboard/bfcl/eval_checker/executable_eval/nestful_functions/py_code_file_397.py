from typing import List



def map_constant(list_of_keys: List[str], constant_value: int) -> dict:

    """Generates a dictionary that maps a list of keys to a given constant value.



    Args:

        list_of_keys: A list of keys to be used in the dictionary.

        constant_value: The constant value to be mapped to each key in the dictionary.

    """

    return dict(zip(list_of_keys, [constant_value] * len(list_of_keys)))

