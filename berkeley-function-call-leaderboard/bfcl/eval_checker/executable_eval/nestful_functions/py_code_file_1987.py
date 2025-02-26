from typing import List



def numbers_within_k(numbers: List[float], threshold: int, k: int) -> List[float]:

    """

    Returns a list of numbers that are within `k` of the threshold.



    Args:

        numbers: A list of numbers.

        threshold: The threshold value.

        k: The maximum difference between a number and the threshold.

    """

    assert all(isinstance(num, (int, float)) for num in numbers), 'Each number in the list must be an integer or a float.'

    assert isinstance(threshold, int), 'The threshold must be an integer.'

    return list(filter(lambda num: abs(num - threshold) <= k, numbers))

