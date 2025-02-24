from typing import Tuple



def read_bit(i: int, position: int) -> int:

    """Reads the bit value at the specified position in the binary representation of an integer.



    Args:

        i: The integer to read the bit from.

        position: The position of the bit to read.



    Returns:

        The bit value at the specified position.

    """

    return (i >> position) & 1



def write_bit(i: int, position: int, value: int) -> int:

    """Writes a new bit value at the specified position in the binary representation of an integer.



    Args:

        i: The integer to write the bit to.

        position: The position of the bit to write.

        value: The new bit value to write at the position.



    Returns:

        The integer with the new bit value written at the specified position.

    """

    mask = 1 << position

    return (i & ~mask) | (value << position)

