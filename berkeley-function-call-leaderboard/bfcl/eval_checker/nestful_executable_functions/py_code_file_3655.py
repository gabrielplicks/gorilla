from typing import Union



def check_input_and_sum(a: Union[int, float], b: Union[int, float]) -> Union[int, float, str]:

    """Calculates the sum of two numbers if they are valid numbers.



    Args:

        a: The first number.

        b: The second number.



    Returns:

        The sum of the two numbers if they are valid numbers, otherwise an error message.

    """

    if not (isinstance(a, int) or isinstance(a, float) or isinstance(b, int) or isinstance(b, float)):

        return "Input is not a number, please provide a valid number."



    return a + b

