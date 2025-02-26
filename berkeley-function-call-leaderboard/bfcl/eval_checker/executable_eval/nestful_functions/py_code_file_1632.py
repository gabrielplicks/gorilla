from typing import Union



def greater_of_two_numbers(num1: int, num2: int, equal_value: Union[int, None]) -> Union[int, None]:

    """Returns the greater of two numbers, or a third value if they are equal.



    Args:

        num1: The first number.

        num2: The second number.

        equal_value: The value to return if the two numbers are equal.



    Returns:

        The greater of the two numbers, or the equal_value if they are equal.

    """

    if num1 > num2:

        return num1

    elif num2 > num1:

        return num2

    else:

        return equal_value

