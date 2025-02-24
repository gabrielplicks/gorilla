from typing import List



def convert_keys(keys: List[str]) -> List[str]:

    """Converts a list of string keys in "dot.separated.name" format to "flat_name" format.

    The function also converts the value of the keys to lowercase.

    Args:

        keys: A list of string keys in "dot.separated.name" format.

    Returns:

        A list of string keys in "flat_name" format.

    """

    return [

        '_'.join(map(str.lower, key.split('.')))

        for key in keys

    ]

