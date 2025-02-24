import numpy as np



def compute_mean_variance_stddev(data: Union[List[float], np.ndarray]) -> Tuple[float, float, float]:

    """Computes the mean, variance, and standard deviation of a given list of numbers.



    Args:

        data: A list or numpy array of numbers.



    Returns:

        A tuple containing the mean, variance, and standard deviation of the input data.

    """

    if isinstance(data, list):

        data = np.array(data)

    elif data.size == 0:

        raise ValueError("Array has no elements")

    mean = np.mean(data)

    variance = np.var(data)

    stddev = np.std(data)



    return mean, variance, stddev

