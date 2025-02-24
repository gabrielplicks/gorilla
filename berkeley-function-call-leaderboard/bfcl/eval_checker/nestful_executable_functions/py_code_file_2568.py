import math



def is_perfect_cube(num: int) -> bool:

    """Determines if a given number is a perfect cube.



    A perfect cube is an integer that can be expressed as the cube of another integer.

    For example, 8 is a perfect cube because it can be expressed as 2^3.



    Args:

        num: The number to check.



    Returns:

        True if the input is a perfect cube, False otherwise.

    """

    cube_root = math.pow(num, 1 / 3)

    if cube_root.is_integer():

        return True

    return False

