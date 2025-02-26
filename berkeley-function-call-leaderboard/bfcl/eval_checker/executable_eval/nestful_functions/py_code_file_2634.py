from typing import Optional



def division_with_remainder(A: int, B: int) -> Optional[int]:

    """Divides A by B and returns the quotient if the remainder is zero.

    Returns None if the remainder is not zero.

    Args:

        A: The dividend.

        B: The divisor.

    """

    if A % B != 0:

        return None

    else:

        return A // B

