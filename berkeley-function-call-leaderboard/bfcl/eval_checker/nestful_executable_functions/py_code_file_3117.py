from typing import Union



def int_to_base_n(num: int, n: Union[int, float]) -> str:

    """Converts a positive integer `num` to a string representing `num` in the given base `n`.



    Args:

        num: A positive integer.

        n: A base between 2 and 16.



    Raises:

        AssertionError: If `num` is not a positive integer or `n` is not between 2 and 16.

    """

    assert (

        num >= 0 and n >= 2 and n <= 16

    ), "Invalid arguments: num must be a positive integer, and n must be between 2 and 16"

    digits = '0123456789ABCDEF'

    result = ''

    while num > 0:

        result = digits[num % n] + result

        num //= n

    return result

