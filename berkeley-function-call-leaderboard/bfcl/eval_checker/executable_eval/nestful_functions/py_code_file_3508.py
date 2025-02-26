from typing import Tuple



def get_area_and_perimeter(height: float, width: float) -> Tuple[float, float]:

    """Calculates the area and perimeter of a rectangle with the given dimensions.



    Args:

        height: The height of the rectangle.

        width: The width of the rectangle.



    Returns:

        A tuple containing the area and perimeter of the rectangle.

    """

    area = height * width

    perimeter = 2 * (height + width)

    return area, perimeter

