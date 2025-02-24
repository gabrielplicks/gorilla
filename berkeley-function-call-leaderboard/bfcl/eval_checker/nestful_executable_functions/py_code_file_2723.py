from typing import List



def swap_items(items: List[Any], i: int, j: int) -> None:

    """

    Swaps the items at indices `i` and `j` in `items`. Raises an `IndexError` if either of the indices is out of bounds.



    Args:

        items: The list of items to swap.

        i: The index of the first item to swap.

        j: The index of the second item to swap.

    """

    if i >= len(items) or j >= len(items):

        raise IndexError("Index out of bounds.")



    if i == j:

        return



    items[i], items[j] = items[j], items[i]

