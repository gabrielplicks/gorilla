import math



def area_of_triangle_from_sides(a: float, b: float, c: float) -> float:

    """Calculates the area of a triangle given the lengths of its sides using Heron's formula.



    Args:

        a: The length of the first side of the triangle.

        b: The length of the second side of the triangle.

        c: The length of the third side of the triangle.

    """

    s = (a + b + c) / 2  # semiperimeter

    area = math.sqrt(s * (s - a) * (s - b) * (s - c))

    return area

