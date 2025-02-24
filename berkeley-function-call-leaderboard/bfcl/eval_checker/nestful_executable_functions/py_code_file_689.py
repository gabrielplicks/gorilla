from typing import List



def extend(left: List, right: List) -> None:

    """Extends `left` with the elements of `right`.



    Args:

        left: The list to be extended.

        right: The list containing elements to be added to `left`.



    Raises:

        TypeError: If `left` or `right` is not a list.

    """

    if not isinstance(left, list) or not isinstance(right, list):

        raise TypeError("Input must be lists")



    if len(right) > 0:

        left.extend(right)

