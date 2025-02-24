from typing import Union



def shape_area(l: int, w: Union[int, None] = None) -> int:

    """Calculates the area of a shape whose width and length are represented by `l` and `w`.

    If the length and width are equal, the shape is a square. If `w` is not provided, it should default to `l`.

    Args:

        l: The length of the shape.

        w: The width of the shape. If not provided, defaults to `l`.

    """

    if w is None:

        return l ** 2

    else:

        return l * w

