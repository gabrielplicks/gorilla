import math



def sum_infinite_series() -> tuple[float, float, float]:

    """Computes the exact sum of the infinite series Σ(1/2^n) and its partial sums.



    Args:

        None



    Returns:

        A tuple containing the exact sum, the partial sum, and the fsum sum.

    """

    r = 0.5

    exact_sum = 2

    partial_sum = 0

    n = 100



    for i in range(1, n + 1):

        partial_sum += r**(i-1)

    fsum_sum = math.fsum([r**(i-1) for i in range(1, n + 1)])



    return exact_sum, partial_sum, fsum_sum

