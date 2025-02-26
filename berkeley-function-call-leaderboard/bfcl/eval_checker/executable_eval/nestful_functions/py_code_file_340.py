import numpy as np

from typing import Tuple



def check_point_on_segment(p1: Tuple[float, float, float], p2: Tuple[float, float, float], p: Tuple[float, float, float]) -> bool:

    """Checks if a point is on a line segment.



    Args:

        p1: The first endpoint of the line segment.

        p2: The second endpoint of the line segment.

        p: The point to check.



    Returns:

        A boolean indicating whether the point is on the line segment.

    """

    if np.linalg.norm(p1 - p) + np.linalg.norm(p2 - p) == np.linalg.norm(p1 - p2):

        return True

    else:

        return False

