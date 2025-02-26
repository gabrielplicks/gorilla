from typing import List, Tuple



def newton_interpolation(points: List[Tuple[float, float]], x: float) -> float:

    """Evaluates a polynomial function at a given point x using Newton’s interpolation.



    Args:

        points: A list of (x, y) points.

        x: The desired x value.

    """

    diffs = []

    for i in range(len(points)):

        diff = points[i][1]

        for j in range(i):

            diff = (diff - diffs[j]) / (points[i][0] - points[j][0])

        diffs.append(diff)

    poly = diffs[0]

    for i in range(1, len(diffs)):

        fact = 1

        for j in range(i):

            fact *= x - points[j][0]

        poly += diffs[i] * fact



    return poly

