from typing import List



def compute_cumulative_sum(nums: List[float]) -> List[float]:

    """Calculates the cumulative sum of a list of numbers.



    Args:

        nums: A list of numbers.



    Returns:

        A new list of the same length as `nums`, where the i-th element is the sum of the first i + 1 elements from `nums`.

    """

    return [sum(nums[:i+1]) for i in range(len(nums))]

