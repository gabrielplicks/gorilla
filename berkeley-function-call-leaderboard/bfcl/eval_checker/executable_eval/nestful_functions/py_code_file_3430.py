from typing import List, Tuple



def count_and_list_ones(grid: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:

    """Counts the number of 1s in a grid and returns their coordinate positions.



    Args:

        grid: A two-dimensional list of 0s and 1s.



    Returns:

        The number of 1s and the list of their coordinates.

    """

    count = 0

    coordinates = []

    for i, row in enumerate(grid):

        for j, value in enumerate(row):

            if value == 1:

                count += 1

                coordinates.append((i, j))

    return count, coordinates

