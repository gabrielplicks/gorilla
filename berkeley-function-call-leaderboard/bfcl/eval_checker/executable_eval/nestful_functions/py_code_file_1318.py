from typing import List



def sliding_window(l: List[int], k: int = 2) -> List[List[int]]:

    """Implements a sliding window on a list.



    Args:

        l: The input list.

        k: The window size. Default is 2.



    Returns:

        A list of slices that represent the sliding window.

    """

    slices = []

    for i in range(len(l) - k + 1):

        slices.append(l[i:i+k])

    return slices

