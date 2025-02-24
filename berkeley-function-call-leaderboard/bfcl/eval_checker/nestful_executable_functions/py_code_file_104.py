from typing import List



def double_list(lst: List[float]) -> List[float]:

    """Returns a list of the original list's values multiplied by 2.



    The resulting list should contain the same number of elements as the original list.

    The function should use a list comprehension, but should not use any built-in functions

    like `map` or `filter`.



    Args:

        lst: The input list.

    """

    return [x * 2 for x in lst]

