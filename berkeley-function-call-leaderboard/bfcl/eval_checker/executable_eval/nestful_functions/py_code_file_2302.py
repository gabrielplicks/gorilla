from typing import List



def get_reversed_digits_power(n: int) -> int:

    """Calculates the sum of the digits' powers to the corresponding position in the reverse order.

    Args:

        n: The input integer.

    """

    digits: List[int] = []

    while n > 0:

        n, remainder = divmod(n, 10)

        digits.append(remainder ** (len(digits) + 1))

    return sum(digits)

