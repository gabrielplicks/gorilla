from typing import Union



def min_max_format(min_val: Union[int, float], max_val: Union[int, float]) -> str:

    """Formats min_val and max_val as a string in the format "{min_val:02}-{max_val:02}".



    Args:

        min_val: The minimum value.

        max_val: The maximum value.



    Raises:

        Exception: If min_val or max_val is not an integer.

    """

    if not isinstance(min_val, int) or not isinstance(max_val, int):

        raise Exception("Inputs must be integers")

    return f"{min_val:02}-{max_val:02}"

