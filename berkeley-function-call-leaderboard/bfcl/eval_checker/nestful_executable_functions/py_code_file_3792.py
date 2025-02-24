from typing import Union



def calculate_summation(a: Union[int, float], b: Union[int, float], step: Union[int, float] = 1) -> Union[int, float]:

    """Calculates the summation of all the numbers in the range [a, b].



    Args:

        a: The starting number of the range.

        b: The ending number of the range.

        step: The step size used in the range. Default is 1.



    Returns:

        The summation of all the numbers in the range [a, b].

    """

    if a > b:

        return 0

    num_steps = (b - a) // step + 1

    summation = 0

    for i in range(a, b + 1, step):

        summation += i

    return summation

