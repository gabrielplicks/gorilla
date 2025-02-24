from typing import List



def calculate_digital_root(n: int) -> int:

    """Calculates the digital root of a given number.



    Args:

        n: The number to calculate the digital root of.



    Returns:

        The digital root of the given number.

    """

    if n < 10:

        return n

    digits: List[int] = [int(digit) for digit in str(n)]

    result: int = sum(digits)



    if result < 10:

        return result

    else:

        return calculate_digital_root(result)

