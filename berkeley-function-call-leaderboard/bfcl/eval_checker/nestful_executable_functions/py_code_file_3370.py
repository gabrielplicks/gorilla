from typing import List



def min_max_scale(nums: List[float]) -> List[float]:

    """Normalizes an array of numbers using min-max scaling.



    Args:

        nums: The input array of numbers.



    Returns:

        The normalized array of numbers.

    """

    min_val = min(nums)

    max_val = max(nums)

    scaled_nums = []



    for num in nums:

        scaled_num = (num - min_val) / (max_val - min_val)

        scaled_nums.append(scaled_num)



    return scaled_nums

