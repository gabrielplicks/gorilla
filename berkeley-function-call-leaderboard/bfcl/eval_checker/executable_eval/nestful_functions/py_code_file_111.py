from typing import Tuple



def idx3d(x: int, y: int, z: int, n: int) -> int:

    """Determines the index of a 3D array element in a 1D array.



    Args:

        x: The x coordinate of the 3D array element.

        y: The y coordinate of the 3D array element.

        z: The z coordinate of the 3D array element.

        n: The dimension of the 3D array.



    Returns:

        The index of the 3D array element in the 1D array representation.

    """

    return x + y * n + z * n * n

