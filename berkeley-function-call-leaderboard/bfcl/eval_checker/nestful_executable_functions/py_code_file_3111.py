from typing import Union



def get_status(value1: Union[int, float], value2: Union[int, float]) -> str:

    """

    Returns a string based on the relationship between two values.

    Args:

        value1: The first value to compare.

        value2: The second value to compare.

    """

    if value1 == value2:

        return "equal"

    elif value1 > value2:

        return "bigger"

    return "smaller"

