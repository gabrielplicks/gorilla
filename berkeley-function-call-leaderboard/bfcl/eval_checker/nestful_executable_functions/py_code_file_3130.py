from typing import List



def get_top_k_union(nums1: List[int], nums2: List[int], k: int) -> List[int]:

    """

    Returns the union of the top `k` numbers from each array, sorted in descending order.



    Args:

        nums1: The first array of numbers.

        nums2: The second array of numbers.

        k: The number of top numbers to return.



    Returns:

        The union of the top `k` numbers from each array, sorted in descending order.

    """

    return sorted(set(nums1) | set(nums2), reverse=True)[:k]

