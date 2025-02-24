from typing import List, Tuple



def find_element(array: List[List[int]], element: int) -> Tuple[int, int]:

    """Finds the index of the first occurrence of a specific element in a 2D list.



    Args:

        array: The 2D list to search.

        element: The element to find.



    Returns:

        A tuple containing the indices of the row and column where the element is found.

        If the element is not found, returns `(-1, -1)`.

    """

    for i, row in enumerate(array):

        for j, col in enumerate(row):

            if col == element:

                return (i, j)

    return (-1, -1)

