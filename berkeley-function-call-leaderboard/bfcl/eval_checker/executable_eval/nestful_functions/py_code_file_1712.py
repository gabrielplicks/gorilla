from typing import List



def get_all_subsets(arr: List[int]) -> List[List[int]]:

    """Computes all subsets of a given set of positive integers stored in a list.



    Args:

        arr: A list of positive integers.



    Returns:

        A list of all subsets, where each subset is represented as a list of integers.

    """

    all_subsets = []



    num_bits = len(arr)

    num_subsets = 2 ** num_bits



    for i in range(num_subsets):

        subset = [arr[j] for j in range(num_bits) if (i >> j) & 1]

        all_subsets.append(subset)



    return all_subsets

