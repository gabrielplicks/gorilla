from typing import List, Tuple



def get_xy(arr: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:

    """Extracts the "x" and "y" coordinates from the given array of tuples.



    Args:

        arr: A list of tuples, where each tuple contains two integers representing the "x" and "y" coordinates.



    Returns:

        A tuple containing two lists. The first list contains the "x" coordinates, and the second list contains the "y" coordinates.

    """

    if not arr:

        return ([], [])

    return ([x for (x, y) in arr], [y for (x, y) in arr])

