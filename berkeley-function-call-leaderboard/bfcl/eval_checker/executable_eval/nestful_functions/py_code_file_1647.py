from typing import Union



def find_max_3(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> Union[int, float]:

    """Finds the largest among 3 numbers.



    Args:

        a: The first number.

        b: The second number.

        c: The third number.

    """

    return max(max(a, b), c)



assert find_max_3(1, 2, 3) == 3

assert find_max_3(5, 2, 1) == 5

assert find_max_3(3, 3, 3) == 3

