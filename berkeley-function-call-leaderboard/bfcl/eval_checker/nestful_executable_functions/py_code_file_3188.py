import math



def normalize_point(x: int, y: int) -> tuple[float, float]:

    """Transforms two integer coordinates (x, y) to a unit vector (x', y') on the unit circle.



    Args:

        x: The x-coordinate of the point.

        y: The y-coordinate of the point.



    Raises:

        ValueError: If the input coordinates (x, y) are invalid.

        ValueError: If the normalized coordinates do not lie on the unit circle.

    """

    if x == 0 and y == 0:

        raise ValueError("Invalid input coordinates (x, y): (0, 0)")

    length = math.sqrt(x**2 + y**2)

    x_normalized = x / length

    y_normalized = y / length

    if x_normalized**2 + y_normalized**2 != 1:

        raise ValueError("Normalized coordinates do not lie on the unit circle")



    return x_normalized, y_normalized

