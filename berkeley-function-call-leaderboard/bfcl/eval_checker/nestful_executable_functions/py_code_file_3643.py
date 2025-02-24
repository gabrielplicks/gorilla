import numpy as np



def split_dataset(data: List[List[float]], k: int) -> Dict[int, np.ndarray]:

    """Splits a dataset into K equal-sized folds and converts each fold to a NumPy array.



    Args:

        data: A list of lists representing the dataset. Each inner list is a 4D vector.

        k: An integer representing the number of desired folds.



    Returns:

        A dictionary with K keys from 0 to K-1, each mapping to a NumPy array representing

        the corresponding fold.

    """

    fold_size = len(data) // k

    folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(k)]

    np_arrays = [np.array(fold) for fold in folds]

    return {i: np_arrays[i] for i in range(k)}

