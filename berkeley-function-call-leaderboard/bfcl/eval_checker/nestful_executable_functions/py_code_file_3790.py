from typing import Any, List



def find_object_index(array: List[Any], object: Any) -> int:

    """Finds the index of an object in an array.



    Args:

        array: The array to search.

        object: The object to find.



    Returns:

        The index of the object in the array if found, otherwise `-1`.

    """

    for i, element in enumerate(array):

        if element == object:

            return i

    return -1

