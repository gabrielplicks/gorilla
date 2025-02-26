from typing import Tuple



def multiply_without_operator(num1: int, num2: int) -> int:

    """Calculates the product of two numbers without using the multiplication operator.



    Args:

        num1: The first number.

        num2: The second number.



    Returns:

        The product of the two numbers.

    """

    if num1 == 0 or num2 == 0:

        return 0

    is_negative = (num1 < 0) ^ (num2 < 0)

    num1 = abs(num1)

    num2 = abs(num2)

    product = 0

    for i in range(num2):

        product += num1



    return -product if is_negative else product

