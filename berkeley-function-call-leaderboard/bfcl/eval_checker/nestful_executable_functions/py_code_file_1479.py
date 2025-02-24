from typing import List, Union



def contains_value(lst: List[Union[int, float]], value: Union[int, float]) -> int:

    """Checks whether a list of numbers contains a specific value.

    If this value exists in the list, returns the index of its first occurrence.

    If it doesn't exist, returns -1.

    Args:

        lst: The list of numbers.

        value: The value to search for.

    """

    if value in lst:

        return lst.index(value)

    return -1

