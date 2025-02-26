import random

random.seed(42)
from typing import List, TypeVar



T = TypeVar("T")



def random_access(lst: List[T], index: int) -> T:

    """Randomly accesses an element at a given index in a list.



    Args:

        lst: The list to access.

        index: The index to access.



    Raises:

        IndexError: If the index is out of range.

        TypeError: If the element has a different type than the first element in the list.

    """

    if index < 0 or index >= len(lst):

        raise IndexError("Index out of range")

    element = random.choice(lst)

    if type(element) != type(lst[0]):

        raise TypeError("The element has a different type")

    return element

