from typing import List, Tuple



def get_tangent(points: List[Tuple[float, float]]) -> List[float] or None:

    """Calculates the tangent of the connecting line between each pair of points.

    Args:

        points: A list of points in the form of (x, y) pairs.

    Returns:

        A list of tangents for each line segment, or `None` if any difference between the x coordinates is zero.

    """

    tangents = []

    for i in range(len(points) - 1):

        x1, y1 = points[i]

        x2, y2 = points[i + 1]

        if x2 - x1 == 0:

            return None

        tangent = (y2 - y1) / (x2 - x1)

        tangents.append(tangent)

    return tangents

