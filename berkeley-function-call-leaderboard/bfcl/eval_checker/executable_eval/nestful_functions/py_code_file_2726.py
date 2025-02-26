from typing import List, Any



def search_element_in_array(array: List[Any], element: Any) -> int:

    """Searches for a given element in an array.

    If the element is not found, the function raises a ValueError with the message "Element not found."

    If the element is found, the function returns the index of the element.

    The function exits early if the element is not found to avoid unnecessary computations.

    Args:

        array: The array to search in.

        element: The element to search for.

    """

    for i, item in enumerate(array):

        if item == element:

            return i

    raise ValueError("Element not found.")

