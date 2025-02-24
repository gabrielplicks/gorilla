import statistics

from typing import List



def calculate_mean_and_stddev(array: List[float]) -> Tuple[float, float]:

    """Calculates the mean and standard deviation of an array of numbers.



    Args:

        array: A list of numbers.



    Returns:

        A tuple containing the mean and standard deviation of the array.

    """

    mean = statistics.mean(array)

    stddev = statistics.stdev(array)

    return mean, stddev

