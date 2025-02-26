from typing import List, Optional



def array_element_at_index(array: List[int], index: int) -> Optional[int]:

    """Returns the element in an array at a given index, or None if the index is out of bounds.



    Args:

        array: A list of integers.

        index: The index of the element to return.

    """

    return array[index] if index < len(array) else None

