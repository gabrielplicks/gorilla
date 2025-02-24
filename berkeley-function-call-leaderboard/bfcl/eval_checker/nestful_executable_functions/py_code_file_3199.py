from typing import Dict



def increment_or_add(dictionary: Dict[str, int], key: str) -> None:

    """Increments the value of a key in a dictionary by 1 if the key exists, or adds the key with a value of 1 if it doesn't.



    Args:

        dictionary: The dictionary to modify.

        key: The key to check or add.

    """

    if key in dictionary:

        dictionary[key] += 1

    else:

        dictionary[key] = 1

