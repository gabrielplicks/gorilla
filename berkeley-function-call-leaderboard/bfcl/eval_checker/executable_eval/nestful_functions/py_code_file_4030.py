from typing import List, Tuple



def compute_centroid(rectangle: List[Tuple[float, float]]) -> List[float]:

    """Calculates the centroid of a rectangle.

    Args:

        rectangle: A list of four points, where each point is a list of two numbers mapping the x and y coordinates of the corresponding point.

    """

    x_avg = sum(point[0] for point in rectangle) / len(rectangle)

    y_avg = sum(point[1] for point in rectangle) / len(rectangle)

    return [x_avg, y_avg]

