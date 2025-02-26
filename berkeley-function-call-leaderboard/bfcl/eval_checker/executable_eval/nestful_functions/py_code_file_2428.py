from typing import List, Tuple



def expand_relative_coordinates(coordinates: List[Tuple[int, int, int]], scope: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:

    """Converts a list of relative coordinates into a list of absolute coordinates.



    Args:

        coordinates: A list of relative coordinates, with each coordinate in the format (x, y, z).

        scope: The size of the coordinate system.



    Returns:

        A list of absolute coordinates, with each coordinate in the format (x_abs, y_abs, z_abs).

    """

    absolute_coordinates = []

    for coordinate in coordinates:

        x_abs = scope[0] + coordinate[0]

        y_abs = scope[1] + coordinate[1]

        z_abs = scope[2] + coordinate[2]

        absolute_coordinates.append((round(x_abs), round(y_abs), round(z_abs)))



    return absolute_coordinates

