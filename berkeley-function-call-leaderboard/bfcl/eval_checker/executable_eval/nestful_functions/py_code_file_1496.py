from typing import Union



def isclose(a: Union[int, float], b: Union[int, float], tolerance: Union[int, float] = 1e-9) -> bool:

    """Determines whether two numbers are close in value.



    Args:

        a: The first number.

        b: The second number.

        tolerance: The maximum difference between `a` and `b` for them to be considered close.

            Defaults to 1e-9 if not provided.



    Returns:

        True if the absolute value of the difference between `a` and `b` is less than or equal to `tolerance`,

        and False otherwise.

    """

    return abs(a - b) <= tolerance

