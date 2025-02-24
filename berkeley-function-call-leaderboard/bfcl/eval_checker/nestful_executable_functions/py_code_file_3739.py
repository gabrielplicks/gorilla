import numpy as np

import numpy.random



def get_random_tensor(seed: int, rank: int) -> np.ndarray:

    """Generates a tensor with a random shape of `rank` dimensions using `numpy.random.rand`.



    Args:

        seed: The seed to set the random seed to.

        rank: The number of dimensions for the tensor.



    Returns:

        A tensor with a random shape of `rank` dimensions.

    """

    np.random.seed(seed)

    shape = np.random.randint(1, 10, size=rank)

    return np.random.rand(*shape)

