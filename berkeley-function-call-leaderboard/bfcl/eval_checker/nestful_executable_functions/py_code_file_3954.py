from typing import Dict



def calc_struct_area(struct: Dict[str, float]) -> int:

    """Calculates the area of a struct with members width and height.

    If either of the members is a float, round down the result to the nearest integer.

    Otherwise, return the result as a float.

    Args:

        struct: A dictionary with keys "width" and "height".

    """

    width = struct["width"]

    height = struct["height"]

    if isinstance(width, float) or isinstance(height, float):

        return int(width * height)

    else:

        return width * height

