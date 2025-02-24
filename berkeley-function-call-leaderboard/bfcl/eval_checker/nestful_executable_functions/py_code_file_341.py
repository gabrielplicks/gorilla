from typing import Union



def multiply_by_2(x: Union[int, float]) -> Union[int, float]:

    """Multiplies the input by 2.

    Args:

        x: The input value.

    """

    return x * 2



def divide_by_3(x: Union[int, float]) -> Union[int, float]:

    """Divides the input by 3.

    Args:

        x: The input value.

    """

    return x / 3



def f(x: Union[int, float]) -> Union[int, None]:

    """Multiplies the input by 2 and then divides it by 3.

    If the input is not a positive integer, then the function returns None.

    Args:

        x: The input value.

    """

    if not isinstance(x, int) or x <= 0:

        return None



    return divide_by_3(multiply_by_2(x))

