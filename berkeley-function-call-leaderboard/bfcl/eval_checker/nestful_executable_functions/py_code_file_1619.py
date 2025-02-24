from typing import Union



def exponentiation(base_num: Union[int, float], exp_power: int) -> Union[int, float]:

    """Calculates the exponentiation of a given integer without the use of built-in exponentiation operators or functions.

    Args:

        base_num: The base number.

        exp_power: The exponent power.

    """

    result = 1

    for _ in range(exp_power):

        result *= base_num

    return result

