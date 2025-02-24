from typing import Union



def mystery_function(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:

    """Calculates the result of a + a^b / b.



    Args:

        a: The first number.

        b: The second number.

    """

    return a + a ** b / b

