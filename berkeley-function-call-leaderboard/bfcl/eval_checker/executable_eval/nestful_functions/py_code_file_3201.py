from typing import List



def remove_none_elements(input_list: List[object]) -> List[object]:

    """Removes all `None` elements from a list using list comprehension.

    Args:

        input_list: The input list.

    Returns:

        A new list with all `None` elements removed.

    """

    return [element for element in input_list if element is not None]

