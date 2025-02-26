from math import factorial



def approximate_e(num: int) -> float:

    """Calculates an approximation of e by summing the first num terms of the infinite series.



    Args:

        num: The number of terms to sum.



    Returns:

        A float representing the approximate value of e.

    """

    sum = 0

    for k in range(num):

        sum += 1 / factorial(k)

    return sum

