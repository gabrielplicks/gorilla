import numpy as np



def sample_skewness(numbers: List[float]) -> float:

    """Calculates the sample skewness of a collection of numbers.



    The sample skewness is defined as the third standardized moment, where the second

    moment (variance) is used to standardize the data.



    Args:

        numbers: A list of numbers.



    Returns:

        The sample skewness as a float.

    """

    mean = np.mean(numbers)

    variance = np.var(numbers, ddof=1)

    skewness = np.mean([(x - mean) ** 3 for x in numbers]) / (variance ** 1.5)



    return skewness

