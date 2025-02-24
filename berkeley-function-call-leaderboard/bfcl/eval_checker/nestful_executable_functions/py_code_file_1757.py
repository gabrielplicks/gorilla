def get_rectangle_area(x1: int = 0, y1: int = 0, x2: int = 1, y2: int = 1) -> Union[int, str]:

    """

    Returns the area of a rectangle with the given coordinates.

    If the coordinates are not valid, returns "Invalid rectangle."

    Args:

        x1: The x-coordinate of the first corner of the rectangle.

        y1: The y-coordinate of the first corner of the rectangle.

        x2: The x-coordinate of the second corner of the rectangle.

        y2: The y-coordinate of the second corner of the rectangle.

    """

    if x1 < 0 or y1 < 0:

        return "Invalid rectangle."

    width = x2 - x1

    height = y2 - y1



    return width * height

