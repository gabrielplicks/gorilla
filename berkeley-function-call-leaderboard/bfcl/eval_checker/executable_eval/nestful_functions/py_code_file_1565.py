from typing import Union



def integer_part(x: Union[int, float]) -> int:

    """Returns the integer part of a float number without using the built-in function math.floor.



    Args:

        x: The float number.

    """

    return int(x * 1)

