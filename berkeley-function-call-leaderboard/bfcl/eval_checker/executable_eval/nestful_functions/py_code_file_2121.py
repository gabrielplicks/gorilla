from typing import Union



def bit_manip_mimic(x: Union[int, float]) -> Union[int, float]:

    """Mimics the behavior of the original bit_manip function using bitwise operations.



    Args:

        x: The input value.



    Returns:

        The result of the bitwise AND operation with the expression `2**8 - 1` to mask the most significant 8 bits of the input `x`.

    """

    return x & ((1 << 8) - 1)

