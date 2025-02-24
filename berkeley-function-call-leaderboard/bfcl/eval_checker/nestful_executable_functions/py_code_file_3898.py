import numpy as np



def compute_mean_tensor_reduction_numpy(x: np.ndarray) -> float:

    """Computes the mean across all axes of a tensor using numpy.



    Args:

        x: The tensor to compute the mean of.

    """

    return np.mean(x)

