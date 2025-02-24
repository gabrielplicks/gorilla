from typing import List



def generate_divisibility_strings(n: int, divisors: List[int]) -> List[str]:

    """Generates a list of strings in the format "n is divisible by x" where n is an integer and x is a divisor of n.

    Args:

        n: The integer to check for divisibility.

        divisors: A list of divisors to check.

    """

    strings = []



    for divisor in divisors:

        if n % divisor == 0:

            strings.append(f"{n} is divisible by {divisor}")



    return strings

