from typing import List



def stack_app(L: List[int]) -> None:

    """Appends a list to itself twice in-place.



    Args:

        L: The list to be appended.

    """

    L += L * 2

    return

