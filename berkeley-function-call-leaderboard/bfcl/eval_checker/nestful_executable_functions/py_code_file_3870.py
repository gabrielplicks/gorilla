from typing import Union



def cyclic_slice(obj: Union[list, str], n: int) -> Union[list, str]:

    """

    Returns a cyclic slice of the object's elements starting at index n.

    The cyclic slice includes the elements from index n and on until the end of the object is reached,

    followed by the elements from the beginning of the object up to index n-1.

    If the index n is out of bounds, the cyclic slice is empty.



    Args:

        obj: The list or string to be sliced.

        n: The starting index of the cyclic slice.



    Returns:

        The cyclic slice of the object's elements starting at index n.

    """

    obj_length = len(obj)

    if not (0 <= n < obj_length):

        return []

    return obj[n:] + obj[:n]

