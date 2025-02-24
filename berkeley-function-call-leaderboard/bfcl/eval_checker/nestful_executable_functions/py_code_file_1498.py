import logging

from typing import Union

null_handler = logging.NullHandler()

logging.basicConfig(handlers=[null_handler])



def log_factorial(n: Union[int, float]) -> Union[int, float]:

    """Calculates the factorial of a non-negative integer `n` (n!).

    If the input `n` is negative, returns -1 and logs the error message "Input is negative".

    Args:

        n: The input number.

    """

    if n < 0:

        logging.error("Input is negative")

        return -1

    result = 1

    for i in range(1, int(n) + 1):

        result *= i

    return result

