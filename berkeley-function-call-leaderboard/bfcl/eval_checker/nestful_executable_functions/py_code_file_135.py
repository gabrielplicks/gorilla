import random

random.seed(42)
from typing import List



def generate_samples(lower_bound: float, upper_bound: float, sample_size: int) -> List[float]:

    """Generates random samples from a uniform distribution within a given range.



    Args:

        lower_bound: The lower bound of the distribution.

        upper_bound: The upper bound of the distribution.

        sample_size: The number of samples to generate.



    Returns:

        A list of random samples from the uniform distribution.

    """

    samples = [random.uniform(0, 1) for _ in range(sample_size)]

    range_size = upper_bound - lower_bound

    scaled_samples = [sample * range_size + lower_bound for sample in samples]



    return scaled_samples

