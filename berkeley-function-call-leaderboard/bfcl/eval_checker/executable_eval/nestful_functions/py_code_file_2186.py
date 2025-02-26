import random

random.seed(42)
from typing import List



def generate_normal_distribution(mu: float, sigma: float, n: int) -> List[float]:

    """Creates a normal distribution with mean `mu` and standard deviation `sigma` when given the parameters `mu`, `sigma`, and `n`.

    The `n` parameter determines the number of random samples to draw from the distribution.

    Then, return an array of length `n` containing the values of the random samples.

    Args:

        mu: The mean of the normal distribution.

        sigma: The standard deviation of the normal distribution.

        n: The number of random samples to draw from the distribution.

    """

    try:

        samples = []

        for _ in range(n):

            samples.append(random.normalvariate(mu, sigma))

        return samples

    except ValueError:

        print("Invalid input parameters. Please check the values of mu, sigma, and n.")

        return []

