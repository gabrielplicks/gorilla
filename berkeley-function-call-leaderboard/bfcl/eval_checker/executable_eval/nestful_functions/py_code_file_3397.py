from typing import List



def is_ordered_ascending(data: List[int]) -> bool:

    """Checks if a list of integers is in ascending order.



    Args:

        data: A list of integers.



    Returns:

        True if the list is in ascending order, False otherwise.

    """

    for i in range(len(data) - 1):

        if data[i] > data[i + 1]:  # Check for non-ascending sequence

            return False

    return True

