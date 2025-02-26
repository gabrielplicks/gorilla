from typing import List



def check_existence(lst: List[int], target: int) -> bool:

    """Checks if all integers in the list are less than the target.



    Args:

        lst: The list of integers to check.

        target: The target integer to compare with.

    """

    return all(map(lambda x: x < target, lst))

