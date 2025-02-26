import random

random.seed(42)
from typing import List, Tuple



def create_random_points(n: int) -> List[Tuple[float, float]]:

    """Generates `n` random points in a two-dimensional space.

    The points are represented as tuples containing two floating-point coordinates (x, y),

    where both x and y are in the range of [0, 100].

    Args:

        n: The number of points to generate.

    Returns:

        A list of points.

    """

    points = []

    for _ in range(n):

        x = random.uniform(0, 100)

        y = random.uniform(0, 100)

        point = (x, y)

        points.append(point)

    return points

