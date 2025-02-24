from typing import List



def product_iterative(numbers: List[float]) -> float:

    """Calculates the product of a list of numbers using an iterative method.

    Args:

        numbers: A list of numbers.

    """

    product = 1

    for number in numbers:

        product *= number

    return product

