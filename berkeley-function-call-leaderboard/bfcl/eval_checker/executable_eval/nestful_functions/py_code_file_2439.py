from typing import List



def final_position(moves: List[str]) -> str:

    """Calculates the final position of a dancer given a list of dance moves.

    Args:

        moves: A list of dance moves represented as strings. Each move can be positive or negative.

    """

    position = 0

    for move in moves:

        position += int(move)

    return str(position)

