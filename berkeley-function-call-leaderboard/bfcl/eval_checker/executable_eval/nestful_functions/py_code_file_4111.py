from typing import Tuple



def convert_3d_to_4d_homogeneous(coordinate: Tuple[float, float, float]) -> list:

    """Converts a 3D coordinate tuple in the Euclidean coordinate system to its 4D homogeneous coordinate system representation.

    Args:

        coordinate: The 3D coordinate tuple.

    """

    x, y, z = coordinate

    z = z + 1

    return list((x, y, z, 1))

