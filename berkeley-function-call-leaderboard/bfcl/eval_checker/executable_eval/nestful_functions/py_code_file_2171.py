from typing import List



def product_without_multiplication(numbers: List[int]) -> int:

    """Calculates the product of a list of numbers without using the built-in multiplication operator.

    Args:

        numbers: A list of numbers.

    """

    if not numbers:

        return 1

    elif len(numbers) == 1:

        return numbers[0]

    else:

        return numbers[0] + product_without_multiplication(numbers[1:])

