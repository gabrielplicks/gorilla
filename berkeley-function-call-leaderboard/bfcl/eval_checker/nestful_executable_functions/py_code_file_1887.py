from typing import List



def get_values_at_indices(nums: List[int], index: List[int]) -> List[int]:

    """

    Returns a new list containing the elements from `nums` at the indices specified by `index`.

    If an index is out of range for `nums`, `None` is added to the new list.



    Args:

        nums: A list of integers.

        index: A list of indices.



    Returns:

        A new list containing the elements from `nums` at the indices specified by `index`.

    """

    new_list = []

    for i in index:

        if i < len(nums):

            new_list.append(nums[i])

        else:

            new_list.append(None)

    return new_list

