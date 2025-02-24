import math



def find_minimum_distance(points: list[tuple[float, float]]) -> float:

    """Calculates the minimum distance between any two points in a list of 2D coordinates.

    Args:

        points: A list of 2D coordinates (x, y).

    Returns:

        The minimum distance between any two points.

    """

    if len(points) < 2:

        return 0

    min_dist = math.inf

    for i in range(len(points) - 1):

        for j in range(i + 1, len(points)):

            x1, y1 = points[i]

            x2, y2 = points[j]

            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if dist < min_dist:

                min_dist = dist



    return min_dist

