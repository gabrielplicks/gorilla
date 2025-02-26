from typing import List



def reverse_first_three(lst: List[int]) -> None:

    """Reverses the first three elements of a list if the list has at least three elements.



    Args:

        lst: The list to reverse the first three elements of.

    """

    if len(lst) >= 3:

        lst[:3] = lst[:3][::-1]

