from typing import Union



def extend_num(num1: Union[int, float], num2: Union[int, float]) -> Union[int, float]:

    """

    Overload the default `+` operator between two numbers.

    Args:

        num1: The first number.

        num2: The second number.

    Returns:

        The extended number.

    """

    if isinstance(num1, int) and isinstance(num2, int):

        return round(num1 + num2)  # Round the sum to the nearest integer if both numbers are integers

    else:

        return round(num1 + num2, 2)  # Round the sum to a floating-point number with two decimal places

