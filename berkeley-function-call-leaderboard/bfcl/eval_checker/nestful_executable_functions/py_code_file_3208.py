def bitwise_left_rotation(x: int, n: int) -> int:

    """Performs a bitwise left rotation to an unsigned integer.



    Args:

        x: The unsigned integer to be rotated.

        n: The number of bits to rotate.

    """

    return (x << n) & 0xFFFFFFFF

