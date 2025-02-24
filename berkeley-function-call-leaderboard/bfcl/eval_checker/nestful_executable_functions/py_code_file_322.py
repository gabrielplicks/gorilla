from typing import List, Union



def nested_list_sum(nums: List[Union[int, List[int]]]) -> int:

    """Calculates the sum of all numbers in a nested list of integers.



    Args:

        nums: A nested list of integers.

    """

    if not nums:

        return 0



    head, *tail = nums

    if isinstance(head, int):

        return head + nested_list_sum(tail)

    else:

        return nested_list_sum(head) + nested_list_sum(tail)

