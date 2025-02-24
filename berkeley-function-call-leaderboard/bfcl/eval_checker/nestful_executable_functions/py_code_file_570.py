from typing import List, Tuple



def is_solvable(grid: List[List[int]]) -> bool:

    """Determines if a sliding puzzle is solvable given a 2-D grid.



    Args:

        grid: A 2-D grid representing the sliding puzzle.



    Returns:

        True if the puzzle is solvable, False otherwise.

    """

    flattened_grid = [tile for row in grid for tile in row if tile != 0]

    sorted_grid = sorted(flattened_grid)

    num_inversions = sum(tile > sorted_grid[i] for i, tile in enumerate(flattened_grid))

    return num_inversions % 2 == 0

