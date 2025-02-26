from typing import Union



def integer_to_boolean(value: Union[int, float]) -> bool:

    """Converts an integer value to a boolean. If the number is odd, the function returns True. If the number is even, the function returns False. If the number is not an integer, the function raises an exception.



    Args:

        value: The input value to be converted to a boolean.



    Raises:

        ValueError: If the input value is not an integer.

    """

    if not isinstance(value, int):

        raise ValueError("Input must be an integer")

    return True if value % 2 == 1 else False

