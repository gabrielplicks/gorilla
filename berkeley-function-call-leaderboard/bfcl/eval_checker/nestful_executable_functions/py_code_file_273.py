from typing import List



def by_index(keys: List[int], values: List[str], i: int) -> str:

    """

    Returns the element of `values` at the same index as the element of `keys` that is equal to `i`.

    If `i` isn't found in `keys`, returns `None`.



    Args:

        keys: A list of integers.

        values: A list of strings.

        i: A positive integer.

    """

    if i not in keys:

        return None

    return [value for key, value in zip(keys, values) if key == i][0]

