import random

random.seed(42)
from typing import List



def bernoulli_sample(n: int, p: float) -> List[int]:

    """Generates N samples of a Bernoulli random variable with parameter p.



    Args:

        n: The number of samples to generate.

        p: The parameter of the Bernoulli distribution.



    Returns:

        A list of N samples.

    """

    samples = []

    for _ in range(n):

        random_num = random.random()

        sample = 1 if random_num < p else 0

        samples.append(sample)

    return samples

