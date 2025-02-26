from typing import Dict, Tuple



def get_diff(d1: Dict[str, int], d2: Dict[str, int]) -> list[Tuple[str, int]]:

    """

    Returns a list of (key, value) pairs present in `d1` but not in `d2`.

    The function maintains the insertion order of the key-value pairs in `d1`.



    Args:

        d1: The first dictionary.

        d2: The second dictionary.

    """

    result = []

    for key, value in d1.items():

        if key not in d2:

            result.append((key, value))

    return result

