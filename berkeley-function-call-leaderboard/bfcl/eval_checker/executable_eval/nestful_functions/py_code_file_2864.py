from typing import List



def create_list_using_loop(start: int, end: int) -> List[int]:

    """Creates a list of integers from start to end, inclusive, that are multiples of 3 and not multiples of 5.

    Args:

        start: The starting integer.

        end: The ending integer.

    """

    result = []

    for i in range(start, end + 1):

        if i % 3 == 0 and i % 5 != 0:

            result.append(i)

    return result

