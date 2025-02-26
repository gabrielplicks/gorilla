from typing import Union



def expected_value(n: Union[int, float]) -> float:

    """Computes the expected value of a discrete uniform random variable X that takes values 1, 2, 3, ..., n, each with probability 1/n.



    Args:

        n: The number of possible outcomes.

    """

    return sum(x * (1/n) for x in range(1, int(n)+1))

