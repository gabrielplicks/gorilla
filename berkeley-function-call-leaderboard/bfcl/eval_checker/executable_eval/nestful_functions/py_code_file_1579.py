from typing import List, Tuple



class Point:

    def __init__(self, x: float, y: float):

        self.x = x

        self.y = y



def find_closest(point: Point, collection: List[Point]) -> Point:

    """

    Finds the closest object to the given point from a list of objects.



    Args:

        point: The given point.

        collection: The list of objects.



    Returns:

        The closest object to the given point.

    """

    min_distance = float("inf")

    closest_object = None

    for obj in collection:

        x1, y1 = point.x, point.y

        x2, y2 = obj.x, obj.y

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        if distance < min_distance:

            min_distance = distance

            closest_object = obj

    return closest_object

