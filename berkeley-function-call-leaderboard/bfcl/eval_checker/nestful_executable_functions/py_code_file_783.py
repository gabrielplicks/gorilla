from typing import List



def add_if_unique(L: List[int], x: int) -> None:

    """Adds `x` to the list `L` if and only if `x` does not already belong to `L`.



    Args:

        L: The list to which `x` may be added.

        x: The element to be added to `L`.

    """

    if x not in L:

        L.append(x)

