import math



def are_floating_point_numbers_equal(x: float, y: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:

    """Determines if two floating-point numbers `x` and `y` are equal to a specified tolerance value.



    Args:

        x: The first floating-point number.

        y: The second floating-point number.

        rel_tol: The relative tolerance.

        abs_tol: The absolute tolerance.



    Returns:

        `True` if the absolute difference between `x` and `y` is less than or equal to the specified tolerance value and `False` otherwise.

    """

    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)

