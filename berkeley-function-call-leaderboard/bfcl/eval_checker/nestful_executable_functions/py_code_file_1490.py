from typing import List



def calculate_bin_centers(nums: List[float]) -> List[float]:

    """Calculates the bin centers for a histogram given a sorted list of numbers.

    Args:

        nums: A sorted list of numbers.

    """

    bin_centers = [(nums[i] + nums[i-1]) / 2 for i in range(1, len(nums))]

    return bin_centers

