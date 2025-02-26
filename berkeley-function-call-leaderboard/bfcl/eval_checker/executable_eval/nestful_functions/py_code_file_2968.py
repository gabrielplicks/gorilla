from typing import List



def switch_elements(lst: List[int]) -> List[int]:

    """Creates a new list where the first element is the second element of the input list,

    the second element is the first element of the input list, the third element is the

    fourth element of the input list, and so on.



    Args:

        lst: The input list.



    Returns:

        A new list with the elements switched as described.

    """

    result = []

    for i in range(0, len(lst), 2):

        result.append(lst[i + 1])

        result.append(lst[i])

    return result

