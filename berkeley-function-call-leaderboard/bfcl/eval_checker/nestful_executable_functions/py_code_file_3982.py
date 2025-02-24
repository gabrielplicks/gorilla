from typing import List



def divide_dataset(n: int) -> List[List[int]]:

    """Divides a dataset of n people into two subsets of equal size.



    Args:

        n: The size of the dataset.



    Returns:

        A list of two subsets, where the first subset contains the indices of the people in the first subset and the second subset contains the indices of the people in the second subset. If the task is impossible, returns -1.

    """

    if n % 2 != 0:

        return -1

    return [[i for i in range(n) if i % 2 == 0], [i for i in range(n) if i % 2 == 1]]

