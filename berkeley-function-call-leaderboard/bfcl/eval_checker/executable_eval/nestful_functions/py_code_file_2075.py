from typing import List



def repeat_list(lst: List[int], n: int) -> List[int]:

    """Repeats a list `n` times.



    Args:

        lst: The input list.

        n: The number of times to repeat the list.



    Returns:

        A new list that repeats the input list `n` times.

    """

    return [item for item in lst for _ in range(n)]

