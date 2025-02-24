from typing import List



def swap_left_right(lst: List[int]) -> List[int]:

    """Swaps the left and right halves of a list of integers.



    Args:

        lst: The list of integers.



    Returns:

        A new list with the left and right halves swapped.

    """

    n = len(lst)

    new_lst = [lst[n // 2 + i] if i < n // 2 else lst[i - n // 2] for i in range(n)]

    return new_lst

