from typing import Optional, List, Tuple



def get_mean_and_variance(nums: List[float]) -> Optional[Tuple[float, float]]:

    """

    Returns the mean and variance of the list `nums`.

    If `nums` is empty, returns `None`.

    Args:

        nums: A list of numbers.

    """

    if not nums:

        return None



    mean = sum(nums) / len(nums)

    variance = sum((x - mean)**2 for x in nums) / len(nums)

    return mean, variance

