from typing import List, Union



def find_winning_symbol(board: List[List[Union[str, None]]]) -> Union[str, None]:

    """Finds the winning symbol in a tic-tac-toe game.



    Args:

        board: A list of lists representing the tic-tac-toe board. Each sublist

            represents a row of cells in the game. The values in the cells are

            either `'X'`, `'O'`, or `None`.



    Returns:

        The winning symbol (either `'X'` or `'O'`) if there is a winning row in

        the game, or `None` if there's no winner.

    """

    for row in board:

        symbol = row[0]

        if symbol and all(cell == symbol for cell in row):

            return symbol

    return None

