def rectangles_intersect(rect1: Tuple[float, float, float, float], rect2: Tuple[float, float, float, float]) -> bool:

    """Determines whether two rectangles intersect.



    Args:

        rect1: A tuple of 4 floats representing the top left and bottom right points of the first rectangle.

        rect2: A tuple of 4 floats representing the top left and bottom right points of the second rectangle.



    Returns:

        True if the rectangles intersect, False otherwise.

    """

    x1, y1, x2, y2 = rect1

    x3, y3, x4, y4 = rect2

    if x1 <= x4 and x2 >= x3 and y1 <= y4 and y2 >= y3:

        return True

    if x1 == x4 and x2 == x3 and y1 == y4 and y2 == y3:

        return True



    return False

