from typing import List



def extract_duplicates(nums: List[int]) -> List[int]:

    """Creates a new list `extracted` that contains all integers from `nums` that are present in `nums` more than once.

    Sorts `extracted` in ascending order.

    Args:

        nums: A list of integers.

    Returns:

        A list of integers that appear more than once in `nums`, sorted in ascending order.

    """

    count = {}

    for num in nums:

        if num in count:

            count[num] += 1

        else:

            count[num] = 1

    extracted = []

    for num, c in count.items():

        if c > 1:

            extracted.append(num)

    extracted.sort()

    return extracted

