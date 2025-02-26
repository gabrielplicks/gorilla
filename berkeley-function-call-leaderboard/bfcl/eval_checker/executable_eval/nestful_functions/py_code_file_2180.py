import random

random.seed(42)
from typing import List



def sample_from_distribution(n: int, p: List[float]) -> List[int]:

    """Generates `n` samples from a probability distribution `p`.



    Args:

        n: The number of samples to generate.

        p: A probability distribution represented by a list of floats.



    Raises:

        ValueError: If the probability distribution is not normalized.

    """

    if sum(p) != 1.0:

        raise ValueError("Probability distribution is not normalized.")



    samples = []

    for _ in range(n):

        r = random.random()

        cumulative_sum = 0.0

        for i, prob in enumerate(p):

            cumulative_sum += prob

            if cumulative_sum >= r:

                samples.append(i)

                break



    return samples

