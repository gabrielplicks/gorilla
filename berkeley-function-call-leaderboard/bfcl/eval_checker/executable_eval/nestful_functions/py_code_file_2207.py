from typing import List



def sum_of_elements(lst: List[int]) -> int:

    """Computes the sum of all elements in a list using recursion.



    Args:

        lst: The input list of integers.



    Returns:

        The sum of all elements in the list.

    """

    def helper(lst: List[int], idx: int) -> int:

        """Recursive helper function to traverse the list and compute the sum.



        Args:

            lst: The input list of integers.

            idx: The current index in the list.



        Returns:

            The sum of elements from the current index to the end of the list.

        """

        if idx == len(lst):

            return 0

        else:

            return lst[idx] + helper(lst, idx + 1)



    return helper(lst, 0)

