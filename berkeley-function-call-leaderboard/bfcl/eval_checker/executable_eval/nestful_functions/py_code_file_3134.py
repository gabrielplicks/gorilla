from typing import List



def find_topmost_cell(grid: List[List[str]]) -> str:

    """Finds the topmost cell in the grid that can be accessed without obstacles.



    Args:

        grid: A 2D grid of characters.



    Returns:

        The topmost accessible cell in the grid, or None if no accessible cell is found.

    """

    for row in reversed(grid):  # Iterate through each row in reverse order

        for cell in row:  # Iterate through each cell in the current row

            if cell != '#':  # Check if the current cell is accessible

                return cell  # Return the first accessible cell we encounter

    return None  # If no accessible cell is found, return None

