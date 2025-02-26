from typing import List



def multiply_max_k_vector(nums: List[int], k: int) -> int:

    """

    Returns the maximum product of any two distinct elements in `nums` that are `k` indices apart.



    Args:

        nums: A non-empty list of non-zero integers.

        k: A positive integer.



    Returns:

        The maximum product of any two distinct elements in `nums` that are `k` indices apart.

    """

    max_product = float('-inf')

    for i in range(len(nums)):

        max_value = max(nums[j] for j in range(max(i - k, 0), min(i + k + 1, len(nums))))

        if i != nums.index(max_value):

            max_product = max(max_product, nums[i] * max_value)

    return max_product

