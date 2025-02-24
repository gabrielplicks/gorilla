from typing import Tuple



def transform_bbox(python_bbox: Tuple[float, float, float, float]) -> list:

    """Transforms a Python bounding box to MATLAB format.

    Args:

        python_bbox: The Python bounding box, a tuple (x, y, w, h).

    """

    x, y, w, h = python_bbox

    x2 = x + w

    y2 = y + h

    matlab_bbox = [x, y, x2, y2]



    return matlab_bbox

