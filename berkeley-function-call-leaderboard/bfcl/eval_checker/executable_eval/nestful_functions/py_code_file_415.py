from typing import List, Tuple



def are_ships_safe(cells: List[List[str]], ships: List[List[Tuple[int, int]]]) -> bool:

    """Check if all the ships are safe and do not overlap with each other on a battleship game board.



    Args:

        cells: The game board represented as a grid of cells.

        ships: The list of ships, where each ship is represented as a list of cells occupied by it.



    Returns:

        True if all the ships are safe, False otherwise.

    """

    for ship in ships:

        for cell in ship:

            for other_ship in ships:

                if ship is not other_ship and cell in other_ship:

                    return False

    return True

