from typing import List



def generate_subsets(nums: List[int]) -> List[List[int]]:

    """

    Generates all possible combinations of the given `nums` as a list of lists.

    The returned subsets are sorted by the sum of the numbers in the subset

    and do not contain duplicate numbers.

    Args:

        nums: A list of integers.

    """

    nums = sorted(set(nums))



    def generate(index: int, subset: List[int]):

        if index == len(nums):

            result.append(subset)

            return

        generate(index + 1, subset + [nums[index]])

        generate(index + 1, subset)



    result = []

    generate(0, [])

    return sorted(result, key=sum)

