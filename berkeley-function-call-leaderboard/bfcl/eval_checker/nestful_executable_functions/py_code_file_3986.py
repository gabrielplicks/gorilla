from typing import List, Tuple



def valid_move(board: List[List[int]], action: Tuple[int, int]) -> bool:

    """Checks if a potential move is valid or not.



    Args:

        board: A two-dimensional array representing a grid of cells.

        action: A tuple of integers representing the x and y coordinates of a potential move.



    Returns:

        A boolean indicating whether the move is valid or not.

    """

    if not isinstance(board, list) or not isinstance(board[0], list):

        raise ValueError("Invalid board configuration")



    if not isinstance(action, tuple) or len(action) != 2:

        raise ValueError("Invalid action configuration")



    x, y = action

    if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]):

        return False



    if board[x][y] == 0:

        return False



    return True

