from typing import Any



def hash_integer(i: int, n: int) -> int:

    """Applies a hash function to an integer `i` and returns an integer that is less than `n`.



    The hash function is a combination of bitwise operations, such as shifting and masking.



    Args:

        i: The integer to be hashed.

        n: The maximum value of the hashed integer.



    Returns:

        An integer that is less than `n`.

    """

    i_shifted = i >> 1  # Shift i right by 1 bit

    i_masked = i_shifted & 1  # Apply a mask to i_shifted

    return i_masked % n  # Return the result modulo n

