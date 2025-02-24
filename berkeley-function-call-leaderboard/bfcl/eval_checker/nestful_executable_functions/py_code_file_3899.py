from typing import List



def split_positive_and_negative(nums: List[int]) -> List[List[int]]:

    """Splits a list of integers into two lists, one containing all the positive integers and the other containing all the negative integers.



    Args:

        nums: A list of integers.



    Returns:

        A list of two lists, the first one containing all the positive integers and the second one containing all the negative integers.

    """

    positive = [num for num in nums if num > 0]

    negative = [num for num in nums if num < 0]

    return [positive, negative]

