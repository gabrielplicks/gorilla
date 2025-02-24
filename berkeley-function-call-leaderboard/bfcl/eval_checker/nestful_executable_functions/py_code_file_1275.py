from typing import List



def calculate_subarray_sum(array: List[int], K: int) -> List[int]:

    """Calculates the sum of all subarrays of size K in the given array.

    Args:

        array: The input array of integers.

        K: The size of the subarrays.

    """

    subarray_sum = []

    for i in range(len(array) - K + 1):

        subarray = array[i:i+K]

        subarray_sum.append(sum(subarray))

    return subarray_sum

