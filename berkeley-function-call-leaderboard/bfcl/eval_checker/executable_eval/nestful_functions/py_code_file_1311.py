from typing import List, Tuple



def average_point(points: List[Tuple[int, int]]) -> Tuple[int, int]:

    """Calculates the average point of a list of points.



    Args:

        points: A list of points represented as tuples in the form (x, y).



    Returns:

        The average point of the input points, rounded to the nearest integer.

    """

    return round(sum(x for x, y in points) / len(points)), round(sum(y for x, y in points) / len(points))

