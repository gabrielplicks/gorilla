from typing import Tuple



def translate_chessboard_to_hexagonal(x: int, y: int) -> Tuple[int, int]:

    """Translates a point's (x, y) coordinates on a chessboard to a pair of coordinates (h, v) on the hexagonal board.

    The (x, y) coordinates are based on a chessboard layout where the top-left corner is (0, 0).

    The (h, v) coordinates are based on a hexagonal board layout where the top-left corner is (0, 0).

    Args:

        x: The x-coordinate on the chessboard.

        y: The y-coordinate on the chessboard.

    Returns:

        A tuple containing the translated coordinates (h, v) on the hexagonal board.

    """

    h = 2 * x

    v = y

    return (h, v)

