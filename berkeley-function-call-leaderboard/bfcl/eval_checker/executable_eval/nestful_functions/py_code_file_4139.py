from typing import Tuple



def resize_box(box: Tuple[int, int, int, int], scale: float) -> Tuple[float, float, float, float]:

    """Resizes a box represented as a tuple of 4 integers (x, y, width, height) with the same center as the original box.

    The resized box should be a multiple of the original box's height and width.

    Args:

        box: The original box represented as a tuple of 4 integers (x, y, width, height).

        scale: The scale as a float.

    """

    x, y, width, height = box

    x_center, y_center = x + width / 2, y + height / 2

    width *= scale

    height *= scale

    x = x_center - width / 2

    y = y_center - height / 2

    return (x, y, width, height)

