import numpy as np

import random

random.seed(42)
from typing import List



def find_smallest_index(A: np.ndarray) -> int:

    """Finds the index of the smallest element in a 1-dimensional Numpy array using a randomized algorithm.



    Args:

        A: A 1-dimensional Numpy array of size N.



    Returns:

        The index of the smallest element in the array.

    """

    while True:

        index = random.randint(0, len(A) - 1)

        if all(A[i] >= A[index] for i in range(len(A))):

            return index

