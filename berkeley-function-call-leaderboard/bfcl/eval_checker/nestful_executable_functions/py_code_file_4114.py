from typing import List



def list_of_lists(n: int) -> List[List[int]]:

    """Generates an n x n list of lists, where each element in the outer list is a list of n numbers that sum to n.



    Args:

        n: The size of the n x n list of lists.



    Returns:

        A list of lists of size n x n, where each element of the outer list is a list of n integers that sum to n.

    """

    lists = []



    for i in range(n):

        inner_list = [i + 1] * n

        lists.append(inner_list)



    return lists

