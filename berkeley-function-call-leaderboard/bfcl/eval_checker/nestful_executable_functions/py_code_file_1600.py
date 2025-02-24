from typing import Tuple



def convert_coordinate_to_tile_coordinates(x: int, y: int) -> Tuple[int, int]:

    """Converts a coordinate to tile coordinates.



    The tile coordinates are defined as a pair of integers (tx, ty) where tx is the x-coordinate of the top-left corner of the tile, and ty is the y-coordinate of the top-right corner of the tile.



    Args:

        x: The x-coordinate of the input coordinate.

        y: The y-coordinate of the input coordinate.



    Returns:

        A tuple of integers (tx, ty) representing the tile coordinates.

    """

    tile_size = 1000

    tx = x // tile_size

    ty = y // tile_size

    return tx, ty

