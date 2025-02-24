import numpy as np



def sample_indices(arr: np.ndarray, n: int) -> List[int]:

    """Samples `n` indices from the input numpy array using a uniform distribution.



    Args:

        arr: The input numpy array.

        n: The number of indices to sample.



    Raises:

        ValueError: If the input array has a length less than `n`.

    """

    if len(arr) < n:

        raise ValueError("Array length must be at least n")

    indices = np.random.choice(len(arr), n, replace=False)



    return indices.tolist()

