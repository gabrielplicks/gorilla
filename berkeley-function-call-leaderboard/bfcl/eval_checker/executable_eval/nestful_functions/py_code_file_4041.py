from typing import Union



def get_minimum_number_of_bits(n: Union[int, float]) -> int:

    """Calculates the minimum number of bits required to represent a given number of inputs.



    Args:

        n: The number of inputs.



    Returns:

        The minimum number of bits required to represent `n` inputs.

    """

    power = 0

    while 2 ** power < n:

        power += 1

    return power

