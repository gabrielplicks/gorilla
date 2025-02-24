from typing import Dict



def print_value(dictionary: Dict, key: str) -> None:

    """Prints the value of a key in a dictionary, raising a KeyError if the key is not found.



    Args:

        dictionary: The dictionary to search.

        key: The key to search for.

    """

    try:

        value = dictionary.get(key)

        print(value)

    except KeyError:

        raise KeyError('Key not found in dictionary')

