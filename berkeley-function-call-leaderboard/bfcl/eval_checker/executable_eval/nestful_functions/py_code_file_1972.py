from typing import List



def find_positions(s: str, c: str) -> List[int]:

    """Finds the positions of a given character in a string.

    Args:

        s: The input string.

        c: The target character.

    """

    positions = []

    for i, ch in enumerate(s):

        if ch == c:

            positions.append(i)

    return positions

