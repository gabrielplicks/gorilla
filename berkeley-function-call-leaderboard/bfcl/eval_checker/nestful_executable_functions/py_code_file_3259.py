from collections import Counter

from typing import List, Union



def find_first_unique_int(nums: List[int]) -> Union[int, None]:

    """Identifies the first integer that occurs only once in a list of integers.

    If there are no such integers, returns `None`.

    Args:

        nums: A list of integers.

    """

    counter = Counter(nums)

    for num in nums:

        if counter[num] == 1:

            return num

    return None

