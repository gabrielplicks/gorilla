from typing import List, Tuple



def get_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:

    """Calculates the centroid of a 2D polygon.



    Args:

        polygon: A 2D polygon represented as an array of points in the form of [x, y].



    Returns:

        The centroid of the polygon.

    """

    x_coords = [point[0] for point in polygon]

    y_coords = [point[1] for point in polygon]

    centroid_x = sum(x_coords) / len(x_coords)

    centroid_y = sum(y_coords) / len(y_coords)

    return (centroid_x, centroid_y)

