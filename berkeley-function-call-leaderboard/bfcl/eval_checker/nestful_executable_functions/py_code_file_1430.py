from typing import Union



def multiply_and_add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:

    """Computes the result of (a * b) + (a + b).



    Args:

        a: The first number.

        b: The second number.

    """

    return (a * b) + (a + b)

