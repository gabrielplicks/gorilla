from typing import List



def move_to_front(lst: List[int], value: int) -> None:

    """Moves a value to the front of a list.



    Args:

        lst: The input list.

        value: The value to move to the front of the list.

    """

    if value in lst:

        index = lst.index(value)

        lst[:] = [value] + lst[:index] + lst[index + 1:]



    return None

