from typing import Union



def integer_square_root(n: Union[int, float]) -> int:

    """Calculates the integer square root of a non-negative number.



    The integer square root of a non-negative number `n` is the greatest integer `x`

    whose square is less than or equal to `n`.



    Args:

        n: A non-negative number.



    Returns:

        The integer square root of `n`.

    """

    lo, hi = 0, n

    while lo <= hi:

        mid = lo + (hi - lo) // 2

        sq_mid = mid ** 2

        if sq_mid == n:

            return mid

        elif sq_mid < n:

            lo = mid + 1

        else:

            hi = mid - 1

    return hi

