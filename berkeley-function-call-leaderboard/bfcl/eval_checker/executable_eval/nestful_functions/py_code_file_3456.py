from typing import List



def filter_array_elements(arr: List[int], min_value: int, max_value: int) -> List[int]:

    """Filters an array of integers to include only those elements that fall within a specified range.



    Args:

        arr: The input array of integers.

        min_value: The minimum value of the range.

        max_value: The maximum value of the range.



    Returns:

        A new array containing the filtered elements.

    """

    filtered_elements = []

    for element in arr:

        if min_value <= element <= max_value:

            filtered_elements.append(element)

    return filtered_elements

