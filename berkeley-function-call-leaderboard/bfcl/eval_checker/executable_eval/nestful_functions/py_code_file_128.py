from typing import Union



def hypotenuse_length(a: Union[int, float], b: Union[int, float]) -> float:

    """Calculates the length of the hypotenuse of a right triangle.



    Args:

        a: The length of one side of the right triangle.

        b: The length of the other side of the right triangle.

    """

    return (a**2 + b**2)**0.5

