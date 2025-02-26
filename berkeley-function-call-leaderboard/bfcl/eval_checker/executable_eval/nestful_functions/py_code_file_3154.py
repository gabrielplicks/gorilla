import numpy as np



def parzen_density_estimation(data_set: np.ndarray, point: np.ndarray) -> float:

    """Estimates the density at the given point using the Parzen window method.



    Args:

        data_set: The data set as a NumPy array.

        point: The point as a NumPy array.



    Returns:

        The density estimation.

    """

    dim = data_set.shape[1]

    rbf = np.exp(-np.linalg.norm(data_set - point, axis=1) ** 2 / (2 * dim))

    scaling = 1 / (np.sqrt(dim * 2 * np.pi) ** dim)

    density = scaling * np.sum(rbf)

    return density

