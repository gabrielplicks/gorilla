from typing import List



def sample_variance(sample: List[float]) -> float:

    """Calculates the sample variance of a series of numbers.



    Args:

        sample: A list of numbers.



    Raises:

        ArithmeticError: If the length of the input list is less than 2.



    Returns:

        The sample variance of the input list.

    """

    n = len(sample)



    if n < 2:

        raise ArithmeticError("Sample length must be at least 2.")



    mean = sum(sample) / n

    variance = sum((x - mean)**2 for x in sample) / (n - 1)



    return variance

