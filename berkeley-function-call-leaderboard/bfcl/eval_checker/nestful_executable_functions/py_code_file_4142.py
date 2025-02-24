from typing import List, Tuple



class Point:

    def __init__(self, x: float, y: float):

        """

        A point in 2D space.

        Args:

            x: The x coordinate.

            y: The y coordinate.

        """

        self.x = x

        self.y = y



def find_smallest_point(points: List[Point]) -> Point:

    """

    Finds the point with the smallest x and y coordinates from a list of points.

    Args:

        points: A list of points.

    """

    smallest_point = points[0]

    for point in points:

        if point.x < smallest_point.x and point.y < smallest_point.y:

            smallest_point = point

    return smallest_point

