from typing import List



def check_if_all_elements_are_equal(arr: List[int], target: int) -> bool:

    """Determines if all elements in the array are equal to the target value.



    Args:

        arr: The array to check.

        target: The target value to compare against.



    Returns:

        True if all elements in the array are equal to the target value, False otherwise.

    """

    for element in arr:

        if element != target:

            return False

    return True

