from typing import List, Union



def average_of_top_k(lst: List[float], k: int) -> Union[float, None]:

    """Computes the average of the top-k (k=1, 2, ..., 5) values in a list of floats.

    If the list has fewer than k elements, the function returns `None`.

    Args:

        lst: A list of floats.

        k: The number of top values to consider.

    """

    if len(lst) < k:

        return None

    lst.sort(reverse=True)

    return sum(lst[:k]) / k

