from typing import Union



def compare_variables(a: Union[int, float], b: Union[int, float]) -> str:

    """

    Compares two variables and returns a string describing the comparison result.



    Args:

        a: The first variable to compare.

        b: The second variable to compare.



    Returns:

        A string describing the comparison result.

    """

    return f"{'a is greater than' if a > b else 'a is less than' if a < b else 'a is equal to'} b"

