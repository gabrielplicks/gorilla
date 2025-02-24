from typing import Union



def is_within_range(search_element: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]) -> bool:

    """Checks if a given element is within a specified range, inclusive.



    Args:

        search_element: The element to search for.

        min_value: The minimum value of the range.

        max_value: The maximum value of the range.



    Returns:

        A boolean indicating whether the search element is within the specified range.

    """

    return min_value <= search_element <= max_value

