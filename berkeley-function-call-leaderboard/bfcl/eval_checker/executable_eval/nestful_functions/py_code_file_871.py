from typing import List



def extend_list(original_list: List[int]) -> List[int]:

    """Creates a new list by extending the original list twice.

    The first half of the new list contains the original list items,

    and the second half contains the original list items in reverse order.

    Args:

        original_list: The original list of integers.

    Returns:

        A new list of integers.

    """

    if not original_list:

        return []



    return [item for item in original_list] + [item for item in reversed(original_list)]

