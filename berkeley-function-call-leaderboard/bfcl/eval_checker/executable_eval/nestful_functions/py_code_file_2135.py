from typing import List, Optional



def get_first_pair_with_sum(lst: List[int], target: int) -> Optional[List[int]]:

    """Finds the first pair of numbers in the list whose sum is equal to the target.



    Args:

        lst: A list of numbers.

        target: An integer representing the target sum.



    Returns:

        A list containing the two numbers if a pair is found, or None if no pair is found.

    """

    for i, num in enumerate(lst):

        for j in range(i + 1, len(lst)):

            if num + lst[j] == target:

                return [num, lst[j]]

    return None

