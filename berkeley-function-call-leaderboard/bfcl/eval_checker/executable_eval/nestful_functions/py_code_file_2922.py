from typing import Union



def equal_within_tolerance(x: Union[int, float], y: Union[int, float], abs_tol: float, rel_tol: float) -> bool:

    """

    Checks whether two numbers are equal within a given absolute and relative tolerance.



    Args:

        x: The first number.

        y: The second number.

        abs_tol: The maximum acceptable absolute difference between the numbers.

        rel_tol: The maximum acceptable relative difference between the numbers.



    Returns:

        True if the numbers are equal within the given tolerances, False otherwise.

    """

    abs_diff = abs(x - y)

    rel_diff = abs_diff / max(abs(x), abs(y))

    return abs_diff <= abs_tol and rel_diff <= rel_tol

