from typing import List



def dot_product_3d_vectors(u: List[float], v: List[float]) -> float:

    """

    Calculates the dot product of two 3D vectors `u` and `v`.



    Args:

        u: A 3D vector represented as a list of [x, y, z] coordinates.

        v: A 3D vector represented as a list of [x, y, z] coordinates.

    """

    dot_product = 0

    for i in range(3):

        dot_product += u[i] * v[i]

    return dot_product

