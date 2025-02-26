import numpy as np



def get_weight_matrix(source_data: np.ndarray, target_data: np.ndarray) -> np.ndarray:

    """Calculates the weight matrix between two input arrays.



    Args:

        source_data: A numpy array of shape (samples, features).

        target_data: A numpy array of shape (samples, features).



    Returns:

        A numpy array of shape (features, features) containing the calculated weights.

    """

    source_range = np.ptp(source_data, axis=0)

    target_range = np.ptp(target_data, axis=0)

    weights = target_range / source_range

    return weights

