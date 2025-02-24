from typing import List, Tuple



def compute_slope(data_set: List[Tuple[float, float]]) -> float:

    """Computes the slope of a line that connects two points in a data set.



    Args:

        data_set: A list of x and y values representing the data set.



    Returns:

        The slope of the line connecting the first and last points in the data set.

    """

    for (x1, y1), (x2, y2) in zip(data_set, data_set[1:]):

        return (y2 - y1) / (x2 - x1)

