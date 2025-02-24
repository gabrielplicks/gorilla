import numpy as np



def sample_exponential(mean: float, n_samples: int) -> np.ndarray:

    """Simulates an exponential distribution with a given mean.



    Args:

        mean: The mean value of the distribution.

        n_samples: The number of samples to generate.



    Returns:

        A list of random numbers that follow an exponential distribution with the given mean.

    """

    uniform_samples = np.random.rand(n_samples)

    exponential_samples = -mean * np.log(1 - uniform_samples)



    return exponential_samples

