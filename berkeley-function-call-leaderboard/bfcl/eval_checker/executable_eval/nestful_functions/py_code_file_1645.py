from typing import List



class NotEnoughItemsException(Exception):

    """Exception raised when the list does not have enough items."""

    pass



def first_k_items(L: List[int], k: int) -> List[int]:

    """Returns a list containing the first k items of L.



    If L does not have enough items, a custom exception NotEnoughItemsException is raised.



    Args:

        L: The input list.

        k: The number of items to return.



    Raises:

        NotEnoughItemsException: If L does not have enough items.

    """

    if len(L) < k:

        raise NotEnoughItemsException("The list does not have enough items.")



    return L[:k][:]  # Copying the first k items of L

