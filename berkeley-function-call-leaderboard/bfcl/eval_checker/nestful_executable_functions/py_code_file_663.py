from typing import List, Tuple



def polygon_orientation(polygon: List[Tuple[float, float]]) -> int:

    """Determines the orientation of a polygon.



    Args:

        polygon: A list of tuples representing the vertices of the polygon. Each tuple contains the x and y coordinates of a vertex.



    Returns:

        1 if the polygon is clockwise, -1 if the polygon is counterclockwise, and 0 if the polygon is degenerate.

    """

    signed_area = 0

    for i in range(len(polygon)):

        j = (i + 1) % len(polygon)

        x1, y1 = polygon[i]

        x2, y2 = polygon[j]

        signed_area += (x1 * y2 - x2 * y1)

    if signed_area > 0:

        return 1

    elif signed_area < 0:

        return -1

    else:

        return 0

