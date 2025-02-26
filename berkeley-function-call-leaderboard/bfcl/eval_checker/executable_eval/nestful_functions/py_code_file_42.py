from typing import Union



def larger_number(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:

    """Returns the larger number of the two input numbers.



    Args:

        a: The first input number.

        b: The second input number.

    """

    if a > b:

        return a

    elif a < b:

        return b

    else:

        return a

