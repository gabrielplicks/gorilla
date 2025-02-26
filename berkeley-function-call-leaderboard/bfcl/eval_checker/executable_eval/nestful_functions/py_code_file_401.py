from typing import Union



def int_to_int(n: Union[int, float]) -> int:

    """Converts an integer or float to an integer.



    Args:

        n: The input number.



    Returns:

        The converted integer.

    """

    sign = -1 if n < 0 else 1

    n_string = str(abs(n))

    int_sum = 0

    for i, c in enumerate(n_string):

        int_sum += (ord(c) - 48) * (10 ** (len(n_string) - 1 - i))



    return sign * int_sum

