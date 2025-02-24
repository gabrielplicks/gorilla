from typing import Union



def sum_with_type_check(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of two numbers, and raises an error if the types of the numbers are not consistent.



    Args:

        x: The first number.

        y: The second number.



    Raises:

        Exception: If the types of x and y are not consistent.

    """

    if type(x) != type(y):

        raise Exception("Types of x and y must be consistent")

    return x + y

