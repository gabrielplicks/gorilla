from typing import List



def find_k_max(nums: List[int], k: int) -> List[int]:

    """Finds the k largest elements from the list in descending order.



    Args:

        nums: A list of numbers.

        k: The number of largest elements to return.



    Returns:

        A list of the k largest elements from the input list in descending order.

    """

    return sorted(nums, reverse=True)[:k]

