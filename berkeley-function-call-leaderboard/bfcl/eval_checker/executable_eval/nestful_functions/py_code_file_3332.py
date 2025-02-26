import numpy as np



def get_numpy_range(start: float, stop: float, step: float = 1) -> np.ndarray:

    """Creates a numpy array of numbers from start to stop by step.



    Args:

        start: The starting number of the range.

        stop: The ending number of the range.

        step: The difference between each number in the range. Default is 1.

    """

    if step == 1:

        step = None

    return np.arange(start, stop, step)

