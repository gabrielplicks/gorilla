from typing import Union



def divide_with_result_type(numerator: int, denominator: int) -> Union[int, float, str]:

    """Divides the numerator by the denominator and returns the result as an integer, float, or string.



    Args:

        numerator: The number to be divided.

        denominator: The number to divide by.



    Returns:

        The result of the division as an integer, float, or string.

    """

    if denominator == 0:

        return 'Second number cannot be zero'



    result = numerator / denominator

    if result.is_integer():

        return int(result)

    else:

        return float(result)

