from typing import List



def filter_list_by_reference(array: List[int], reference_array: List[int]) -> List[int]:

    """

    Filters a list of elements from `array` that do not appear in `reference_array` using list comprehension and filtering techniques.



    Args:

        array: The list of elements to filter.

        reference_array: The list of elements to use as a reference for filtering.



    Returns:

        A list of elements from `array` that do not appear in `reference_array`.

    """

    return [element for element in array if element not in reference_array]

