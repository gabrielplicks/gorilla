import math



def transform_to_polar(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:

    """Transforms a set of 2D coordinates from Cartesian to polar coordinates.



    Args:

        coordinates: A list of 2D coordinates represented as tuples.



    Returns:

        A list of tuples, where each tuple contains the radial coordinate and the angular coordinate in radians.

    """

    transformed_coordinates = []

    for x, y in coordinates:

        r = math.sqrt(x**2 + y**2)

        theta = math.atan2(y, x)

        transformed_coordinates.append((r, theta))



    return transformed_coordinates

