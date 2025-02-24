from functools import reduce

from typing import Callable



def product_of_even_digits(n: int) -> int:

    """Calculates the product of all even digits in a positive integer `n`.



    Args:

        n: A positive integer.



    Raises:

        ValueError: If `n` is not a positive integer.

    """

    if not isinstance(n, int) or n < 0:

        raise ValueError("Input must be a positive integer.")



    even_digits: Callable[[str], str] = filter(lambda d: int(d) % 2 == 0, str(n))

    product: int = reduce(lambda x, y: int(x) * int(y), even_digits, 1)



    return product

