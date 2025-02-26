from typing import List



def int_to_bool_list(num: int) -> List[bool]:

    """Converts an integer to a boolean list representing its binary representation.

    Each boolean represents the corresponding bit of the binary representation of the integer,

    starting from the least significant bit (LSB) and ending with the most significant bit (MSB).

    Args:

        num: The integer to convert.

    """

    bitmask = 1

    result = []

    while bitmask <= num:

        if num & bitmask != 0:

            result.append(True)

        else:

            result.append(False)

        bitmask <<= 1

    return result

