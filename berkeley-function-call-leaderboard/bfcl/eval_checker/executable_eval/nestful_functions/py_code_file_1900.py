from typing import List



def generate_hex_list(integers: List[int]) -> List[str]:

    """Generates a list of hexadecimal numbers from the given list of integers.

    The hexadecimal numbers are three-character strings, padded with zeros on the left if necessary.

    If the input integer is negative, the corresponding hexadecimal number is preceded by a minus sign (-).

    Args:

        integers: The list of integers.

    Returns:

        A list of hexadecimal numbers.

    """

    return [hex(integer)[2:].zfill(3) if integer >= 0 else f"-{hex(abs(integer))[2:].zfill(3)}" for integer in integers]

