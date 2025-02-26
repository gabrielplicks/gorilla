from typing import Union



def add_or_divide(a: Union[int, float], b: Union[int, float], div: bool = False) -> Union[int, float]:

    """Determines whether to add or divide two numbers `a` and `b`.



    If `a` and `b` are both integers, returns their sum. If `div` is set to `True`, returns `a / b`.

    Otherwise, returns `a + b`.



    Args:

        a: The first number.

        b: The second number.

        div: If set to `True`, returns `a / b`. Otherwise, returns `a + b`.

    """

    if div:

        return a / b

    elif isinstance(a, int) and isinstance(b, int):

        return a + b

    else:

        return a + b

