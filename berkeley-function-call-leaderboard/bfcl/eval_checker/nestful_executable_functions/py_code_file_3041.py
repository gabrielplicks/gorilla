from typing import Union



def assign_value_to_y(condition_string: str, x: int) -> Union[int, str]:

    """Assigns a value to a variable based on a condition.



    Args:

        condition_string: A string representing the condition.

        x: An integer value.



    Returns:

        The value assigned to the variable `y`.

    """

    condition = eval(condition_string)

    if condition:

        y = 1

    else:

        y = 'x is not greater than 3'

    return y

