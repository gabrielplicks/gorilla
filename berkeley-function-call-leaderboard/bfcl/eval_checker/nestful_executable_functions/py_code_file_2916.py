from typing import Optional



def check_in_range(x: float, low: Optional[float] = None, high: Optional[float] = None) -> bool:

    """

    Checks if a value `x` is in the range [`low`, `high`]. If `low` is not specified, assign it the value of `None`,

    and if `high` is not specified, assign it the value of `None`. If both `low` and `high` are `None`, return `True`.



    Args:

        x: The value to check.

        low: The lower bound of the range.

        high: The upper bound of the range.

    """

    if low is None and high is None:

        return True

    if low is None:

        if x <= high:

            return True

        else:

            return False

    if high is None:

        if x >= low:

            return True

        else:

            return False

    if low <= x <= high:

        return True

    else:

        return False

