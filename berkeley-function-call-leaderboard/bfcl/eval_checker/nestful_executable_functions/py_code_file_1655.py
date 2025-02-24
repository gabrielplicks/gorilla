from typing import Union



def multiply_without_star(a: Union[int, float], b: int) -> Union[int, float]:

    """Multiplies two numbers without using the `*` operator.



    Args:

        a: The first number to multiply.

        b: The second number to multiply.

    """

    product = 0

    for _ in range(b):

        product += a

    return product

