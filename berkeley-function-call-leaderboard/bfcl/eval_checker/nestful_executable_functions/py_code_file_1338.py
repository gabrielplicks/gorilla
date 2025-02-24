from typing import List



def windowed_mean(nums: List[float], n: int) -> List[float]:

    """Calculates the mean of each window of `n` consecutive elements in `nums`.



    Args:

        nums: A list of numbers.

        n: The size of the window.



    Returns:

        A list of mean values for each window.

    """

    return [sum(nums[i:i+n]) / n for i in range(len(nums) - n + 1)]

