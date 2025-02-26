from typing import Tuple



def is_member_of_coset(x: int) -> bool:

    """Determines whether a number `x` is a member of a given coset of the form `x = a * 2**k + b` where `a`, `b`, and `k` are non-negative integers.



    Args:

        x: The number to check.



    Returns:

        True if `x` is a member of the coset, False otherwise.

    """

    for a in range(x + 1):

        for b in range(x + 1):

            for k in range(x + 1):

                if x == a * (2 ** k) + b:

                    return True

    return False

