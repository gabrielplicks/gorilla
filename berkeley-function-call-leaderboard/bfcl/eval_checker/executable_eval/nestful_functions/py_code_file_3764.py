import numpy as np

from typing import List



def find_constant_sum_subarrays(arr: np.ndarray, k: int) -> int:

    """Finds the number of subarrays in `arr` that have a sum of `k`.



    Args:

        arr: A 1-D NumPy array of positive integers.

        k: An integer.



    Returns:

        The number of subarrays in `arr` that have a sum of `k`.

    """

    count = 0

    n = len(arr)

    for i in range(n):

        window_sum = 0

        for j in range(i, n):

            window_sum += arr[j]

            if window_sum == k:

                count += 1

                break

            if window_sum > k:

                break

    return count

