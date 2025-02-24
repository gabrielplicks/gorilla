from typing import List



def check_target_presence(nums: List[int], target: int) -> bool:

    """Checks if the target is present in the list.



    Args:

        nums: A list of integers.

        target: The target integer.



    Returns:

        True if the target is found in the list, False otherwise.

    """

    for num in nums:

        if num == target:

            return True

    return False

