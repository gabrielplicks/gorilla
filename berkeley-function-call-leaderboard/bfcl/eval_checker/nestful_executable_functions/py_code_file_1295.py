from typing import List



def harmonic_mean(nums: List[float]) -> float:

    """Computes the harmonic mean of a list of numbers.



    Args:

        nums: A list of numbers.



    Returns:

        The harmonic mean of the numbers in the list. If any number in the list is 0,

        the function returns `inf`.

    """

    if 0 in nums:

        return float("inf")



    total = 0

    for num in nums:

        total += 1 / num



    return len(nums) / total

